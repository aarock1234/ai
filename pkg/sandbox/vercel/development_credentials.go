package vercel

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
)

// DevelopmentCredentialsOptions configure DevelopmentCredentials.
type DevelopmentCredentialsOptions struct {
	// DeviceSignIn signs in when the Vercel CLI has no stored access token.
	// Without a Prompt, a missing token is an error instead.
	DeviceSignIn

	// TeamID and ProjectID scope the credentials. Unset values are
	// inferred.
	TeamID    string
	ProjectID string

	// HTTPClient sends the requests. Nil uses http.DefaultClient.
	HTTPClient *http.Client
}

// DevelopmentCredentials returns credentials for local development, a port
// of the development credentials of @vercel/sandbox. The token is the
// Vercel CLI's access token from auth.json: SignIn obtains one when none is
// stored, and an expired one is refreshed.
//
// An unset team or project is inferred: the project linked in the working
// directory's .vercel/project.json, or else the project
// "vercel-sandbox-default-project", created when missing, in the first team
// that allows it among the given team, the user's default team, the hobby
// team the user owns (preferring the personal one) and the personal
// account.
//
// Like upstream, successful credentials are cached per team and project for
// the life of the process and are not refreshed when the token expires
// later; a failure is not cached.
func DevelopmentCredentials(ctx context.Context, options DevelopmentCredentialsOptions) (Credentials, error) {
	scope := oidcScope{
		teamID:    options.TeamID,
		projectID: options.ProjectID,
	}

	credentials, err := developmentCache.get(ctx, scope, func(ctx context.Context) (Credentials, error) {
		return generateDevelopmentCredentials(ctx, options)
	})
	if err != nil {
		return Credentials{}, err
	}

	options.logger().WarnContext(
		ctx,
		"using inferred Vercel credentials",
		slog.String("team", credentials.TeamID),
		slog.String("project", credentials.ProjectID),
	)

	return credentials, nil
}

// generateDevelopmentCredentials obtains the CLI's access token and infers
// the unset scope.
func generateDevelopmentCredentials(ctx context.Context, options DevelopmentCredentialsOptions) (Credentials, error) {
	httpClient := options.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}

	token, err := developmentAccessToken(ctx, options, httpClient)
	if err != nil {
		return Credentials{}, err
	}

	if options.TeamID != "" && options.ProjectID != "" {
		return Credentials{
			Token:     token,
			TeamID:    options.TeamID,
			ProjectID: options.ProjectID,
		}, nil
	}

	scope, err := inferScope(ctx, scopeInference{
		token:      token,
		teamID:     options.TeamID,
		httpClient: httpClient,
	})
	if err != nil {
		return Credentials{}, err
	}

	if scope.created {
		options.logger().InfoContext(
			ctx,
			"created default Vercel project",
			slog.String("project", scope.projectID),
			slog.String("team", scope.teamID),
		)
	}

	return mergeCredentials(Credentials{
		Token:     token,
		TeamID:    options.TeamID,
		ProjectID: options.ProjectID,
	}, Credentials{
		TeamID:    scope.teamID,
		ProjectID: scope.projectID,
	}), nil
}

// developmentAccessToken returns the CLI's access token from auth.json,
// signing in first when none is stored and refreshing it when it expired.
func developmentAccessToken(ctx context.Context, options DevelopmentCredentialsOptions, httpClient *http.Client) (string, error) {
	configDir := cliConfigDir()

	config, ok := readCLIAuthConfig(configDir)
	if !ok || config.Token == "" {
		if err := signInForDevelopment(ctx, options, httpClient); err != nil {
			return "", err
		}
	}

	return cliAccessToken(ctx, cliAccessTokenRequest{
		configDir:  configDir,
		httpClient: httpClient,
	})
}

func signInForDevelopment(ctx context.Context, options DevelopmentCredentialsOptions, httpClient *http.Client) error {
	if options.Prompt == nil {
		return errors.New("failed to retrieve authentication token; set VERCEL_OIDC_TOKEN, provide a Vercel API token, or sign in with SignIn")
	}

	logger := options.logger()

	logger.WarnContext(ctx, "no Vercel OIDC token found, starting device authorization for development")

	err := SignIn(ctx, SignInOptions{
		DeviceSignIn: options.DeviceSignIn,
		HTTPClient:   httpClient,
	})
	if err != nil {
		logger.ErrorContext(ctx, "Vercel authentication failed", slog.Any("error", err))

		return err
	}

	logger.InfoContext(ctx, "authenticated with Vercel")

	return nil
}

// developmentCache holds the latest development credentials of the process.
var developmentCache = developmentCredentialsCache{
	lock: make(chan struct{}, 1),
}

// developmentCredentialsCache caches credentials for one team and project
// scope. Concurrent callers wait for one generation.
type developmentCredentialsCache struct {
	// lock is a semaphore, so that waiting callers can give up when their
	// context ends.
	lock chan struct{}

	scope       oidcScope
	credentials *Credentials
}

// get returns the cached credentials of scope, or generates and caches
// them. Other scopes replace the cached ones.
func (c *developmentCredentialsCache) get(
	ctx context.Context,
	scope oidcScope,
	generate func(ctx context.Context) (Credentials, error),
) (Credentials, error) {
	select {
	case c.lock <- struct{}{}:
	case <-ctx.Done():
		return Credentials{}, context.Cause(ctx)
	}

	defer func() {
		<-c.lock
	}()

	if c.credentials != nil && c.scope == scope {
		return *c.credentials, nil
	}

	c.credentials = nil

	credentials, err := generate(ctx)
	if err != nil {
		return Credentials{}, err
	}

	c.scope = scope
	c.credentials = &credentials

	return credentials, nil
}
