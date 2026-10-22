package vercel

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// defaultProjectName is the project of development sandboxes when no
// project is linked or given.
const defaultProjectName = "vercel-sandbox-default-project"

// teamsPageSize is the number of teams requested per page.
const teamsPageSize = 20

// accountAPIError is a failed Vercel REST API response during scope
// inference.
type accountAPIError struct {
	StatusCode int
	Message    string
}

func (e *accountAPIError) Error() string {
	return fmt.Sprintf("vercel api request failed with status %d: %s", e.StatusCode, e.Message)
}

// isSkippableTeamError reports a team that cannot be used, so that the next
// candidate is tried.
func isSkippableTeamError(err error) bool {
	apiErr, ok := errors.AsType[*accountAPIError](err)

	return ok && (apiErr.StatusCode == http.StatusPaymentRequired || apiErr.StatusCode == http.StatusForbidden)
}

func isNotFoundError(err error) bool {
	apiErr, ok := errors.AsType[*accountAPIError](err)

	return ok && apiErr.StatusCode == http.StatusNotFound
}

// inferredScope is the team and project of development credentials.
type inferredScope struct {
	teamID    string
	projectID string

	// created reports that the default project was created.
	created bool
}

// scopeInference is a scope lookup with a Vercel access token.
type scopeInference struct {
	token      string
	teamID     string
	httpClient *http.Client
}

// accountUser is the part of the authenticated user that inference reads.
type accountUser struct {
	DefaultTeamID *string `json:"defaultTeamId"`
	Username      string  `json:"username"`
}

// accountTeam is a team of the authenticated user. Teams without the
// required members are skipped.
type accountTeam struct {
	ID         string   `json:"id"`
	Slug       string   `json:"slug"`
	UpdatedAt  *float64 `json:"updatedAt"`
	Membership *struct {
		Role string `json:"role"`
	} `json:"membership"`
	Billing *struct {
		Plan string `json:"plan"`
	} `json:"billing"`
}

func (t accountTeam) isComplete() bool {
	return t.ID != "" && t.Slug != "" && t.Membership != nil && t.Billing != nil
}

func (t accountTeam) isOwnedHobbyTeam() bool {
	return t.Membership.Role == "OWNER" && t.Billing.Plan == "hobby"
}

func (t accountTeam) updatedAt() float64 {
	if t.UpdatedAt == nil {
		return 0
	}

	return *t.UpdatedAt
}

// teamsPage is one page of the user's teams.
type teamsPage struct {
	Teams      []json.RawMessage `json:"teams"`
	Pagination struct {
		Next *int64 `json:"next"`
	} `json:"pagination"`
}

// inferScope resolves the team and project of development credentials, a
// port of @vercel/sandbox's inferScope: the project linked in the working
// directory, or the default project in the given team, or else in the first
// usable team among the user's default team, their best owned hobby team
// and their personal account.
func inferScope(ctx context.Context, inference scopeInference) (inferredScope, error) {
	if linked, ok := readWorkingDirectoryLinkedProject(); ok {
		return inferredScope{
			teamID:    linked.teamID,
			projectID: linked.projectID,
		}, nil
	}

	if inference.teamID != "" {
		return inference.tryTeam(ctx, inference.teamID)
	}

	user, err := inference.fetchUser(ctx)
	if err != nil {
		return inferredScope{}, err
	}

	if user.DefaultTeamID != nil {
		scope, err := inference.tryTeam(ctx, *user.DefaultTeamID)
		if !isSkippableTeamError(err) {
			return scope, err
		}
	}

	scope, found, err := inference.tryHobbyTeams(ctx, user)
	if err != nil || found {
		return scope, err
	}

	scope, err = inference.tryTeam(ctx, user.Username)
	if !isSkippableTeamError(err) {
		return scope, err
	}

	return inferredScope{}, &accountAPIError{
		StatusCode: http.StatusForbidden,
		Message:    fmt.Sprintf("authenticated as %q but none of the available teams allow sandbox creation", user.Username),
	}
}

// tryHobbyTeams tries the best owned hobby team of each page of the user's
// teams, skipping the default team, which was tried already.
func (i scopeInference) tryHobbyTeams(ctx context.Context, user accountUser) (inferredScope, bool, error) {
	var next *int64

	for {
		page, err := i.fetchTeams(ctx, next)
		if err != nil {
			return inferredScope{}, false, err
		}

		best, ok := bestHobbyTeam(decodeTeams(page.Teams), user.Username)
		if ok && !isDefaultTeam(best, user) {
			scope, err := i.tryTeam(ctx, best.ID)
			if !isSkippableTeamError(err) {
				return scope, err == nil, err
			}
		}

		if page.Pagination.Next == nil {
			return inferredScope{}, false, nil
		}

		next = page.Pagination.Next
	}
}

func isDefaultTeam(team accountTeam, user accountUser) bool {
	return user.DefaultTeamID != nil && team.ID == *user.DefaultTeamID
}

// decodeTeams keeps the teams that have every required member.
func decodeTeams(entries []json.RawMessage) []accountTeam {
	var teams []accountTeam

	for _, entry := range entries {
		var team accountTeam
		if err := json.Unmarshal(entry, &team); err != nil || !team.isComplete() {
			continue
		}

		teams = append(teams, team)
	}

	return teams
}

// bestHobbyTeam returns the owned hobby team named after the user, or else
// the most recently updated one.
func bestHobbyTeam(teams []accountTeam, username string) (accountTeam, bool) {
	var best accountTeam

	found := false

	for _, team := range teams {
		if !team.isOwnedHobbyTeam() {
			continue
		}

		if team.Slug == username {
			return team, true
		}

		if !found || team.updatedAt() > best.updatedAt() {
			best = team
			found = true
		}
	}

	return best, found
}

// tryTeam uses the default project of a team, creating it when missing.
func (i scopeInference) tryTeam(ctx context.Context, teamID string) (inferredScope, error) {
	query := teamQuery(teamID)
	scope := inferredScope{
		teamID:    teamID,
		projectID: defaultProjectName,
	}

	err := i.fetch(ctx, accountRequest{
		method:   http.MethodGet,
		endpoint: "/v2/projects/" + url.PathEscape(defaultProjectName) + "?" + query,
	}, nil)
	if err == nil {
		return scope, nil
	}

	if !isNotFoundError(err) {
		return inferredScope{}, err
	}

	err = i.fetch(ctx, accountRequest{
		method:   http.MethodPost,
		endpoint: "/v11/projects?" + query,
		body: map[string]string{
			"name": defaultProjectName,
		},
	}, nil)
	if err != nil {
		return inferredScope{}, err
	}

	scope.created = true

	return scope, nil
}

// teamQuery selects a team by ID or, for other values, by slug.
func teamQuery(teamID string) string {
	if strings.HasPrefix(teamID, "team_") {
		return "teamId=" + url.QueryEscape(teamID)
	}

	return "slug=" + url.QueryEscape(teamID)
}

func (i scopeInference) fetchUser(ctx context.Context) (accountUser, error) {
	var response struct {
		User *accountUser `json:"user"`
	}

	err := i.fetch(ctx, accountRequest{
		method:   http.MethodGet,
		endpoint: "/v2/user",
	}, &response)
	if err != nil {
		return accountUser{}, err
	}

	if response.User == nil || response.User.Username == "" {
		return accountUser{}, errors.New("invalid vercel user response: missing user.username")
	}

	return *response.User, nil
}

func (i scopeInference) fetchTeams(ctx context.Context, until *int64) (teamsPage, error) {
	endpoint := "/v2/teams?limit=" + strconv.Itoa(teamsPageSize)
	if until != nil {
		endpoint += "&until=" + strconv.FormatInt(*until, 10)
	}

	var page teamsPage

	err := i.fetch(ctx, accountRequest{
		method:   http.MethodGet,
		endpoint: endpoint,
	}, &page)
	if err != nil {
		return teamsPage{}, err
	}

	return page, nil
}

// accountRequest is a Vercel REST API request.
type accountRequest struct {
	method   string
	endpoint string

	// body is encoded as JSON when it is not nil.
	body any
}

// fetch sends a request to the Vercel REST API and decodes a successful
// response into result, when it is not nil. A failure is an
// *accountAPIError.
func (i scopeInference) fetch(ctx context.Context, request accountRequest, result any) error {
	httpRequest, err := i.newRequest(ctx, request)
	if err != nil {
		return err
	}

	response, err := i.httpClient.Do(httpRequest)
	if err != nil {
		return fmt.Errorf("vercel api request %s: %w", request.endpoint, err)
	}

	defer func() {
		_ = response.Body.Close()
	}()

	content, err := io.ReadAll(io.LimitReader(response.Body, maxResponseBodyBytes))
	if err != nil {
		return fmt.Errorf("read vercel api response: %w", err)
	}

	if !isSuccessStatus(response.StatusCode) {
		return &accountAPIError{
			StatusCode: response.StatusCode,
			Message:    accountErrorMessage(content),
		}
	}

	if result == nil {
		return nil
	}

	if err := json.Unmarshal(content, result); err != nil {
		return fmt.Errorf("decode vercel api response %s: %w", request.endpoint, err)
	}

	return nil
}

func (i scopeInference) newRequest(ctx context.Context, request accountRequest) (*http.Request, error) {
	var body io.Reader

	if request.body != nil {
		encoded, err := json.Marshal(request.body)
		if err != nil {
			return nil, fmt.Errorf("encode vercel api request: %w", err)
		}

		body = bytes.NewReader(encoded)
	}

	httpRequest, err := http.NewRequestWithContext(ctx, request.method, defaultBaseURL+request.endpoint, body)
	if err != nil {
		return nil, fmt.Errorf("create vercel api request: %w", err)
	}

	httpRequest.Header.Set("Authorization", "Bearer "+i.token)
	httpRequest.Header.Set("Content-Type", "application/json")
	httpRequest.Header.Set("User-Agent", userAgent)

	return httpRequest, nil
}

// accountErrorMessage formats an API error body as "CODE: message", or
// returns the body when it is not an API error.
func accountErrorMessage(content []byte) string {
	var failure struct {
		Error *struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}

	if err := json.Unmarshal(content, &failure); err != nil || failure.Error == nil || failure.Error.Code == "" {
		return string(content)
	}

	return strings.ToUpper(failure.Error.Code) + ": " + failure.Error.Message
}

// readWorkingDirectoryLinkedProject reads .vercel/project.json in the
// working directory. A missing or incomplete file links no project.
func readWorkingDirectoryLinkedProject() (oidcScope, bool) {
	content, err := os.ReadFile(filepath.Join(".vercel", "project.json"))
	if err != nil {
		return oidcScope{}, false
	}

	var project struct {
		ProjectID *string `json:"projectId"`
		OrgID     *string `json:"orgId"`
	}

	if err := json.Unmarshal(content, &project); err != nil || project.ProjectID == nil || project.OrgID == nil {
		return oidcScope{}, false
	}

	return oidcScope{
		teamID:    *project.OrgID,
		projectID: *project.ProjectID,
	}, true
}
