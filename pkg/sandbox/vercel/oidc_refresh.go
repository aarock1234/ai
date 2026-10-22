package vercel

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"time"
)

// oidcTokenEndpoint issues OIDC tokens for a project to a Vercel access
// token.
const oidcTokenEndpoint = "https://api.vercel.com/v1/projects/"

// vercelIDPattern matches team and project IDs, which are passed to the
// Vercel CLI.
var vercelIDPattern = regexp.MustCompile(`^[A-Za-z0-9_][A-Za-z0-9_-]*$`)

// oidcRefresh describes one token refresh.
type oidcRefresh struct {
	scope      oidcScope
	buffer     time.Duration
	httpClient *http.Client
}

// oidcTokenResponse is a token issued for a project, with the response it
// came in, which is cached as received.
type oidcTokenResponse struct {
	token string
	raw   []byte
}

// refreshOIDCToken returns a fresh token for the project, a port of
// @vercel/oidc's refreshToken: the token cached for the project in the user
// data directory, or a new one obtained with the Vercel CLI's credentials
// and then cached.
func refreshOIDCToken(ctx context.Context, refresh oidcRefresh) (string, error) {
	scope, err := completeOIDCScope(refresh.scope)
	if err != nil {
		return "", err
	}

	cached, err := loadCachedOIDCToken(scope.projectID)
	if err != nil {
		return "", err
	}

	if cached != "" {
		expired, err := isTokenExpired(cached, refresh.buffer)
		if err != nil {
			return "", err
		}

		if !expired {
			return cached, nil
		}
	}

	issued, err := issueOIDCToken(ctx, scope, refresh)
	if err != nil {
		return "", err
	}

	if err := saveCachedOIDCToken(scope.projectID, issued.raw); err != nil {
		return "", err
	}

	return issued.token, nil
}

// issueOIDCToken obtains a new token with the Vercel CLI's credentials:
// through the CLI when it keeps them in the system keyring, and otherwise
// with the access token of its auth.json.
func issueOIDCToken(ctx context.Context, scope oidcScope, refresh oidcRefresh) (oidcTokenResponse, error) {
	configDir := cliConfigDir()

	storage, err := cliCredentialStorage(configDir)
	if err != nil {
		return oidcTokenResponse{}, err
	}

	if storage == credentialStorageKeyring {
		return oidcTokenFromCLI(ctx, scope)
	}

	accessToken, err := cliAccessToken(ctx, cliAccessTokenRequest{
		configDir:  configDir,
		buffer:     refresh.buffer,
		httpClient: refresh.httpClient,
	})
	if err != nil {
		return oidcTokenResponse{}, err
	}

	return exchangeOIDCToken(ctx, refresh.httpClient, accessToken, scope)
}

// completeOIDCScope fills a missing team or project from the linked
// project's .vercel/project.json.
func completeOIDCScope(scope oidcScope) (oidcScope, error) {
	if scope.teamID == "" || scope.projectID == "" {
		linked, err := linkedProject()
		if err != nil {
			return oidcScope{}, err
		}

		if scope.projectID == "" {
			scope.projectID = linked.projectID
		}

		if scope.teamID == "" {
			scope.teamID = linked.teamID
		}
	}

	if scope.projectID == "" {
		return oidcScope{}, errors.New("failed to refresh oidc token: no project specified; try re-linking your project with `vc link`")
	}

	return scope, nil
}

// linkedProject reads .vercel/project.json in the nearest directory, from
// the working directory upward, that contains a .vercel directory.
func linkedProject() (oidcScope, error) {
	root, err := projectRoot()
	if err != nil {
		return oidcScope{}, err
	}

	content, err := os.ReadFile(filepath.Join(root, ".vercel", "project.json"))
	if err != nil {
		return oidcScope{}, fmt.Errorf("project.json not found, have you linked your project with `vc link`?: %w", err)
	}

	var project struct {
		ProjectID string `json:"projectId"`
		OrgID     string `json:"orgId"`
	}

	if err := json.Unmarshal(content, &project); err != nil {
		return oidcScope{}, fmt.Errorf("decode .vercel/project.json: %w", err)
	}

	if project.ProjectID == "" && project.OrgID == "" {
		return oidcScope{}, errors.New("expected a projectId in .vercel/project.json; try running `vc link` to re-link your project")
	}

	return oidcScope{
		teamID:    project.OrgID,
		projectID: project.ProjectID,
	}, nil
}

func projectRoot() (string, error) {
	directory, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("find vercel project root: %w", err)
	}

	for directory != filepath.Dir(directory) {
		if _, err := os.Stat(filepath.Join(directory, ".vercel")); err == nil {
			return directory, nil
		}

		directory = filepath.Dir(directory)
	}

	return "", errors.New("unable to find project root directory; have you linked your project with `vc link`?")
}

// userDataDir is where tokens are cached: XDG_DATA_HOME, or the platform's
// application data directory.
func userDataDir() (string, error) {
	if dataHome := os.Getenv("XDG_DATA_HOME"); dataHome != "" {
		return dataHome, nil
	}

	switch runtime.GOOS {
	case "darwin":
		return homeSubdirectory("Library", "Application Support")

	case "linux":
		return homeSubdirectory(".local", "share")

	case "windows":
		if localAppData := os.Getenv("LOCALAPPDATA"); localAppData != "" {
			return localAppData, nil
		}
	}

	return "", errors.New("unable to find user data directory")
}

func homeSubdirectory(elements ...string) (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("unable to find user data directory: %w", err)
	}

	return filepath.Join(append([]string{home}, elements...)...), nil
}

func cachedOIDCTokenPath(projectID string) (string, error) {
	directory, err := userDataDir()
	if err != nil {
		return "", err
	}

	return filepath.Join(directory, "com.vercel.token", projectID+".json"), nil
}

// loadCachedOIDCToken returns the cached token of the project, or an empty
// string when none is cached.
func loadCachedOIDCToken(projectID string) (string, error) {
	path, err := cachedOIDCTokenPath(projectID)
	if err != nil {
		return "", err
	}

	content, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return "", nil
	}

	if err != nil {
		return "", fmt.Errorf("read cached vercel oidc token: %w", err)
	}

	return decodeOIDCTokenResponse(content)
}

// saveCachedOIDCToken caches a token response, readable only by the user
// and the user's group.
func saveCachedOIDCToken(projectID string, response []byte) error {
	path, err := cachedOIDCTokenPath(projectID)
	if err != nil {
		return err
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o770); err != nil {
		return fmt.Errorf("create vercel oidc token cache: %w", err)
	}

	if err := os.WriteFile(path, response, 0o660); err != nil {
		return fmt.Errorf("cache vercel oidc token: %w", err)
	}

	return os.Chmod(path, 0o660)
}

// decodeOIDCTokenResponse reads the token of a token response.
func decodeOIDCTokenResponse(content []byte) (string, error) {
	var response struct {
		Token *string `json:"token"`
	}

	if err := json.Unmarshal(content, &response); err != nil {
		return "", fmt.Errorf("vercel oidc token is malformed: %w", err)
	}

	if response.Token == nil {
		return "", errors.New("vercel oidc token is malformed: expected a string-valued token property")
	}

	return *response.Token, nil
}

// exchangeOIDCToken requests a token for the project with a Vercel access
// token.
func exchangeOIDCToken(ctx context.Context, httpClient *http.Client, accessToken string, scope oidcScope) (oidcTokenResponse, error) {
	query := url.Values{
		"source": {
			"vercel-oidc-refresh",
		},
	}

	if scope.teamID != "" {
		query.Set("teamId", scope.teamID)
	}

	target := oidcTokenEndpoint + url.PathEscape(scope.projectID) + "/token?" + query.Encode()

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, target, nil)
	if err != nil {
		return oidcTokenResponse{}, fmt.Errorf("create vercel oidc token request: %w", err)
	}

	request.Header.Set("Authorization", "Bearer "+accessToken)

	response, err := httpClient.Do(request)
	if err != nil {
		return oidcTokenResponse{}, fmt.Errorf("failed to refresh oidc token: %w", err)
	}

	defer func() {
		_ = response.Body.Close()
	}()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return oidcTokenResponse{}, fmt.Errorf("failed to refresh oidc token: %s", response.Status)
	}

	content, err := io.ReadAll(io.LimitReader(response.Body, maxErrorBodyBytes))
	if err != nil {
		return oidcTokenResponse{}, fmt.Errorf("read vercel oidc token response: %w", err)
	}

	return newOIDCTokenResponse(content)
}

func newOIDCTokenResponse(content []byte) (oidcTokenResponse, error) {
	token, err := decodeOIDCTokenResponse(content)
	if err != nil {
		return oidcTokenResponse{}, err
	}

	return oidcTokenResponse{
		token: token,
		raw:   content,
	}, nil
}

// oidcTokenFromCLI runs `vercel project token` for credentials kept in the
// system keyring.
//
// Go adaptation: the CLI is looked up in PATH only, not in the
// node_modules/.bin directories that @vercel/cli-exec also searches.
func oidcTokenFromCLI(ctx context.Context, scope oidcScope) (oidcTokenResponse, error) {
	if !isValidCLIScope(scope) {
		return oidcTokenResponse{}, errors.New("failed to refresh oidc token with the vercel cli: invalid project or team id")
	}

	args := []string{
		"project",
		"token",
		scope.projectID,
		"--format=json",
	}

	if scope.teamID != "" {
		args = append(args, "--scope", scope.teamID)
	}

	command := exec.CommandContext(ctx, "vercel", args...)

	output, err := command.Output()
	if err != nil {
		return oidcTokenResponse{}, cliTokenError(err)
	}

	return newOIDCTokenResponse(output)
}

// isValidCLIScope requires IDs that cannot be mistaken for CLI options or
// shell syntax: a project ID and, when set, a team ID.
func isValidCLIScope(scope oidcScope) bool {
	if !vercelIDPattern.MatchString(scope.projectID) {
		return false
	}

	return scope.teamID == "" || vercelIDPattern.MatchString(scope.teamID)
}

// cliTokenError describes a failed `vercel project token`, including its
// standard error.
func cliTokenError(err error) error {
	exitErr, ok := errors.AsType[*exec.ExitError](err)
	if !ok || len(exitErr.Stderr) == 0 {
		return fmt.Errorf("failed to refresh oidc token with the vercel cli: %w", err)
	}

	return fmt.Errorf("failed to refresh oidc token with the vercel cli: %w: %s", err, strings.TrimSpace(string(exitErr.Stderr)))
}
