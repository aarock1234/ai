package claudecode

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
)

// defaultAnthropicBaseURL is the Anthropic API endpoint.
const defaultAnthropicBaseURL = "https://api.anthropic.com"

// credentialEnvironmentVariables name the variables that hold credentials.
var credentialEnvironmentVariables = []string{
	"AI_GATEWAY_API_KEY",
	"ANTHROPIC_API_KEY",
	"ANTHROPIC_AUTH_TOKEN",
	"CLAUDE_CODE_OAUTH_TOKEN",
}

// resolvedMode is the route requests take: directly to Anthropic or through
// the AI Gateway.
type resolvedMode string

// Resolved authentication modes.
const (
	resolvedDirect    resolvedMode = "direct"
	resolvedAIGateway resolvedMode = "ai-gateway"
)

// apiKeyHelperTimeout bounds the apiKeyHelper command.
const apiKeyHelperTimeout = 30 * time.Second

// apiKeyReader returns an API key from a custom source. It reports false
// when the source is not configured or yields no key.
type apiKeyReader func(ctx context.Context) (string, bool, error)

// noAPIKeyHelper reads no API key.
func noAPIKeyHelper(context.Context) (string, bool, error) {
	return "", false, nil
}

// resolveEnv returns the environment Claude Code needs to authenticate with
// Anthropic, directly or through the AI Gateway. An explicit mode pins the
// route; otherwise the AI Gateway is preferred when its key is available.
// readAPIKey is the last-resort source of direct credentials.
func resolveEnv(ctx context.Context, auth harness.Authentication, processEnv map[string]string, readAPIKey apiKeyReader) (map[string]string, error) {
	suppliedEnvironment := auth.Environment != nil

	environment := processEnv
	if suppliedEnvironment {
		environment = auth.Environment
	}

	if suppliedEnvironment || auth.Mode == AuthenticationDirect {
		readAPIKey = noAPIKeyHelper
	}

	if !suppliedEnvironment && auth.Mode == AuthenticationDirect {
		return pickAnthropic(ctx, environment, readAPIKey)
	}

	gateway := harness.AIGatewayAuthFromEnv(environment)
	usesGateway := !suppliedEnvironment && auth.Mode == harness.AuthenticationAIGateway
	if usesGateway || gateway.APIKey != "" {
		return pickGateway(gateway), nil
	}

	return pickAnthropic(ctx, environment, readAPIKey)
}

// resolveMode returns the route of the configured authentication.
func resolveMode(auth harness.Authentication, processEnv map[string]string) resolvedMode {
	environment := processEnv

	switch {
	case auth.Environment != nil:
		environment = auth.Environment
	case auth.Mode == AuthenticationDirect:
		return resolvedDirect
	case auth.Mode == harness.AuthenticationAIGateway:
		return resolvedAIGateway
	}

	if harness.AIGatewayAuthFromEnv(environment).APIKey != "" {
		return resolvedAIGateway
	}

	return resolvedDirect
}

// pickAnthropic returns the direct Anthropic credentials of environment,
// falling back to the API key helper for the API key and auth token. The
// helper only runs when one of them is missing.
func pickAnthropic(ctx context.Context, environment map[string]string, readAPIKey apiKeyReader) (map[string]string, error) {
	env := make(map[string]string)

	apiKey, hasAPIKey := bridgeadapter.FirstPresent(environment, "ANTHROPIC_API_KEY")
	authToken, hasAuthToken := bridgeadapter.FirstPresent(environment, "ANTHROPIC_AUTH_TOKEN")

	if !hasAPIKey || !hasAuthToken {
		helperKey, hasHelperKey, err := readAPIKey(ctx)
		if err != nil {
			return nil, err
		}

		if !hasAPIKey && hasHelperKey {
			apiKey = helperKey
		}

		if !hasAuthToken && hasHelperKey {
			authToken = helperKey
		}
	}

	setIfNotEmpty(env, "ANTHROPIC_API_KEY", apiKey)
	setIfNotEmpty(env, "ANTHROPIC_AUTH_TOKEN", authToken)
	setIfNotEmpty(env, "CLAUDE_CODE_OAUTH_TOKEN", environment["CLAUDE_CODE_OAUTH_TOKEN"])
	setIfNotEmpty(env, "ANTHROPIC_BASE_URL", environment["ANTHROPIC_BASE_URL"])

	return env, nil
}

// pickGateway returns the AI Gateway credentials and endpoint.
func pickGateway(gateway harness.AIGatewayAuth) map[string]string {
	env := map[string]string{
		"AI_GATEWAY_BASE_URL": gateway.BaseURL,
		"ANTHROPIC_BASE_URL":  gateway.BaseURL,
	}

	if gateway.APIKey != "" {
		env["AI_GATEWAY_API_KEY"] = gateway.APIKey
		env["ANTHROPIC_API_KEY"] = gateway.APIKey
	}

	return env
}

// claudeSettings is the part of ~/.claude/settings.json the adapter reads.
type claudeSettings struct {
	APIKeyHelper string `json:"apiKeyHelper"`
}

// readAPIKeyHelper runs the apiKeyHelper command of ~/.claude/settings.json,
// as the claude CLI does, so that credentials from a password manager work
// without ANTHROPIC_API_KEY. A missing settings file or helper reads no key.
//
// Go adaptation: upstream ignores unreadable settings and failed helpers.
// This port reports them, and bounds the helper with a timeout.
func readAPIKeyHelper(ctx context.Context) (string, bool, error) {
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return "", false, nil
	}

	settingsPath := filepath.Join(home, ".claude", "settings.json")

	content, err := os.ReadFile(settingsPath)
	if errors.Is(err, fs.ErrNotExist) {
		return "", false, nil
	}

	if err != nil {
		return "", false, fmt.Errorf("read claude code settings: %w", err)
	}

	var settings claudeSettings
	if err := json.Unmarshal(content, &settings); err != nil {
		return "", false, fmt.Errorf("decode claude code settings %s: %w", settingsPath, err)
	}

	if settings.APIKeyHelper == "" {
		return "", false, nil
	}

	key, err := runAPIKeyHelper(ctx, settings.APIKeyHelper)
	if err != nil {
		return "", false, err
	}

	return key, key != "", nil
}

// runAPIKeyHelper runs the helper command with a timeout and returns its
// trimmed output.
func runAPIKeyHelper(ctx context.Context, command string) (string, error) {
	helperCtx, cancel := context.WithTimeout(ctx, apiKeyHelperTimeout)
	defer cancel()

	output, err := exec.CommandContext(helperCtx, "sh", "-c", command).Output()

	if ctx.Err() != nil {
		return "", context.Cause(ctx)
	}

	if helperCtx.Err() != nil {
		return "", fmt.Errorf("claude code apiKeyHelper did not finish within %s", apiKeyHelperTimeout)
	}

	if err != nil {
		return "", fmt.Errorf("run claude code apiKeyHelper: %w", err)
	}

	return strings.TrimSpace(string(output)), nil
}

// requestTransformations returns the transformations that inject the real
// API key and auth token in place of their sandbox placeholders.
func requestTransformations(sources harness.RequestTransformationSources, mode resolvedMode) ([]harness.RequestTransformation, error) {
	matchURL := anthropicMatchURL(sources.Env, mode)

	var transformations []harness.RequestTransformation

	if hasVariable(sources.Env, sources.SandboxEnv, "ANTHROPIC_API_KEY") {
		transformation, err := harness.NewCredentialRequestTransformation(harness.CredentialRequestTransformationOptions{
			MatchURL: matchURL,
			MatchHeaders: map[string]string{
				"x-api-key": sources.SandboxEnv["ANTHROPIC_API_KEY"],
			},
			TransformHeaders: map[string]string{
				"x-api-key": sources.Env["ANTHROPIC_API_KEY"],
			},
		})
		if err != nil {
			return nil, err
		}

		transformations = append(transformations, transformation)
	}

	for _, name := range []string{"ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"} {
		if !hasVariable(sources.Env, sources.SandboxEnv, name) {
			continue
		}

		transformation, err := bearerTransformation(matchURL, sources.SandboxEnv[name], sources.Env[name])
		if err != nil {
			return nil, err
		}

		transformations = append(transformations, transformation)
	}

	return transformations, nil
}

// bearerTransformation replaces a placeholder bearer token.
func bearerTransformation(matchURL string, placeholder string, credential string) (harness.RequestTransformation, error) {
	return harness.NewCredentialRequestTransformation(harness.CredentialRequestTransformationOptions{
		MatchURL: matchURL,
		MatchHeaders: map[string]string{
			"Authorization": "Bearer " + placeholder,
		},
		TransformHeaders: map[string]string{
			"Authorization": "Bearer " + credential,
		},
	})
}

// anthropicMatchURL returns the endpoint the credentials are sent to.
func anthropicMatchURL(env map[string]string, mode resolvedMode) string {
	baseURL, ok := env["ANTHROPIC_BASE_URL"]
	if ok || mode == resolvedAIGateway {
		return baseURL
	}

	return defaultAnthropicBaseURL
}

// hasVariable reports whether both environments define name.
func hasVariable(env map[string]string, sandboxEnv map[string]string, name string) bool {
	_, inEnv := env[name]
	_, inSandboxEnv := sandboxEnv[name]

	return inEnv && inSandboxEnv
}

// setIfNotEmpty sets a variable to a non-empty value.
func setIfNotEmpty(env map[string]string, name string, value string) {
	if value != "" {
		env[name] = value
	}
}
