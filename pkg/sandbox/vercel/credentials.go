package vercel

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"

	"github.com/aarock1234/ai/pkg/harness"
)

// authenticationMessage explains how to configure Vercel Sandbox
// credentials.
const authenticationMessage = "Vercel Sandbox authentication failed. Set VERCEL_OIDC_TOKEN or VERCEL_OIDC_TOKEN_FILE, link the project with `vercel link`, set VERCEL_TOKEN with VERCEL_TEAM_ID and VERCEL_PROJECT_ID, or pass Token, TeamID and ProjectID, then verify that they can access Vercel Sandbox."

// Credentials authenticate Vercel Sandbox API calls. Set all three fields,
// or none to resolve them from the environment.
type Credentials struct {
	// Token is a Vercel access token or OIDC token.
	Token string

	// TeamID is the team that owns the sandboxes.
	TeamID string

	// ProjectID is the project the sandboxes belong to.
	ProjectID string
}

// resolveCredentials returns explicit credentials, or credentials from the
// environment: a Vercel OIDC token, whose claims name the team and project,
// or a VERCEL_TOKEN with VERCEL_TEAM_ID and VERCEL_PROJECT_ID. Fields set
// explicitly take precedence over the environment.
//
// The OIDC token comes from VERCEL_OIDC_TOKEN_FILE or VERCEL_OIDC_TOKEN and
// is refreshed through the linked project when it is missing or expired,
// as @vercel/oidc does. Without an OIDC token in the environment, a
// VERCEL_TOKEN takes precedence over the refresh. When no OIDC token can
// be obtained and DevelopmentSignIn is set, the credentials are
// DevelopmentCredentials.
func resolveCredentials(ctx context.Context, options ConnectionOptions, httpClient *http.Client) (Credentials, error) {
	explicit := options.Credentials

	if explicit.Token != "" && explicit.TeamID != "" && explicit.ProjectID != "" {
		return explicit, nil
	}

	if explicit.Token == "" && prefersOIDC() {
		return resolveOIDCCredentials(ctx, options, httpClient)
	}

	credentials := mergeCredentials(explicit, Credentials{
		Token:     os.Getenv("VERCEL_TOKEN"),
		TeamID:    os.Getenv("VERCEL_TEAM_ID"),
		ProjectID: os.Getenv("VERCEL_PROJECT_ID"),
	})

	if missing := missingCredentials(credentials); len(missing) > 0 {
		return Credentials{}, authenticationError(fmt.Errorf(
			"missing credentials parameters to access the vercel api: %s",
			strings.Join(missing, ", "),
		))
	}

	return credentials, nil
}

// prefersOIDC reports whether credentials come from an OIDC token: one is
// in the environment, or no VERCEL_TOKEN is, so that the token is
// refreshed through the linked project.
func prefersOIDC() bool {
	return hasOIDCEnvironment() || os.Getenv("VERCEL_TOKEN") == ""
}

// resolveOIDCCredentials obtains credentials from an OIDC token, falling
// back to development credentials when DevelopmentSignIn is set.
func resolveOIDCCredentials(ctx context.Context, options ConnectionOptions, httpClient *http.Client) (Credentials, error) {
	explicit := options.Credentials

	credentials, oidcErr := credentialsFromOIDC(ctx, explicit, httpClient)
	if oidcErr == nil {
		return mergeCredentials(explicit, credentials), nil
	}

	if options.DevelopmentSignIn == nil {
		return Credentials{}, authenticationError(oidcErr)
	}

	development, err := DevelopmentCredentials(ctx, DevelopmentCredentialsOptions{
		DeviceSignIn: *options.DevelopmentSignIn,
		TeamID:       explicit.TeamID,
		ProjectID:    explicit.ProjectID,
		HTTPClient:   httpClient,
	})
	if err != nil {
		return Credentials{}, authenticationError(errors.Join(oidcErr, err))
	}

	return development, nil
}

// credentialsFromOIDC obtains a current OIDC token, using the explicit team
// and project to refresh it.
func credentialsFromOIDC(ctx context.Context, explicit Credentials, httpClient *http.Client) (Credentials, error) {
	source := newOIDCTokenSource(httpClient, "")

	scope := oidcScope{
		teamID:    explicit.TeamID,
		projectID: explicit.ProjectID,
	}

	token, err := source.token(ctx, scope, 0)
	if err != nil {
		return Credentials{}, err
	}

	return credentialsFromOIDCToken(token)
}

// credentialsFromOIDCToken reads the team and project claims of an OIDC
// token without verifying it.
func credentialsFromOIDCToken(token string) (Credentials, error) {
	claims, err := decodeOIDCClaims(token)
	if err != nil {
		return Credentials{}, err
	}

	if claims.OwnerID == "" || claims.ProjectID == "" {
		return Credentials{}, errors.New("invalid vercel oidc token: missing owner_id or project_id")
	}

	return Credentials{
		Token:     token,
		TeamID:    claims.OwnerID,
		ProjectID: claims.ProjectID,
	}, nil
}

// mergeCredentials fills the unset explicit fields from fallback.
func mergeCredentials(explicit Credentials, fallback Credentials) Credentials {
	if explicit.Token == "" {
		explicit.Token = fallback.Token
	}

	if explicit.TeamID == "" {
		explicit.TeamID = fallback.TeamID
	}

	if explicit.ProjectID == "" {
		explicit.ProjectID = fallback.ProjectID
	}

	return explicit
}

// missingCredentials names the unset credential fields.
func missingCredentials(credentials Credentials) []string {
	var missing []string

	if credentials.Token == "" {
		missing = append(missing, "token")
	}

	if credentials.TeamID == "" {
		missing = append(missing, "teamId")
	}

	if credentials.ProjectID == "" {
		missing = append(missing, "projectId")
	}

	return missing
}

// authenticationError wraps a credential failure for harness callers.
func authenticationError(cause error) error {
	return &harness.SandboxAuthenticationError{
		Message:           authenticationMessage,
		SandboxProviderID: providerID,
		Cause:             cause,
	}
}
