package harness

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"log/slog"
	"maps"
	"net/url"
	"regexp"
	"slices"
	"strings"
)

// sandboxCredentialPlaceholderPrefix starts every generated credential
// placeholder.
const sandboxCredentialPlaceholderPrefix = "aisdkhc_"

// sandboxCredentialPlaceholderPattern matches a generated placeholder.
var sandboxCredentialPlaceholderPattern = regexp.MustCompile(`^aisdkhc_[A-Za-z0-9_-]{43}$`)

// NewSandboxCredentialPlaceholder returns a random, non-secret placeholder
// to expose in the sandbox instead of a credential that request
// transformations inject outside it.
func NewSandboxCredentialPlaceholder() string {
	random := make([]byte, 32)
	_, _ = rand.Read(random)

	return sandboxCredentialPlaceholderPrefix + base64.RawURLEncoding.EncodeToString(random)
}

// IsSandboxCredentialPlaceholder reports whether value is a generated
// placeholder.
func IsSandboxCredentialPlaceholder(value string) bool {
	return sandboxCredentialPlaceholderPattern.MatchString(value)
}

// CredentialForwardingOptions configure ApplyCredentialForwarding.
type CredentialForwardingOptions struct {
	// Environment is the environment to forward into the sandbox.
	Environment map[string]string

	// CredentialEnvironmentVariables name the variables holding credentials.
	CredentialEnvironmentVariables []string

	// Forwarding customizes each credential. Nil forwards them unchanged.
	Forwarding CredentialForwardingFunc
}

// ApplyCredentialForwarding returns a copy of the environment with each
// credential passed through the forwarding function.
func ApplyCredentialForwarding(ctx context.Context, options CredentialForwardingOptions) (map[string]string, error) {
	forwarded := maps.Clone(options.Environment)
	if forwarded == nil {
		forwarded = make(map[string]string)
	}

	if options.Forwarding == nil {
		return forwarded, nil
	}

	for _, name := range uniqueNames(options.CredentialEnvironmentVariables) {
		credential, ok := forwarded[name]
		if !ok {
			continue
		}

		value, err := options.Forwarding(ctx, CredentialForwardingEvent{
			Credential:              credential,
			EnvironmentVariableName: name,
		})
		if err != nil {
			return nil, fmt.Errorf("forward credential %s: %w", name, err)
		}

		forwarded[name] = value
	}

	return forwarded, nil
}

// SandboxCredentialEnvironmentOptions configure
// ResolveSandboxCredentialEnvironment.
type SandboxCredentialEnvironmentOptions struct {
	Environment                    map[string]string
	CredentialEnvironmentVariables []string
	Forwarding                     CredentialForwardingFunc

	// Previous is the sandbox credential environment of an earlier start of
	// the same session. Its placeholders are reused, so request
	// transformations installed for them keep matching.
	Previous map[string]string
}

// ResolveSandboxCredentialEnvironment returns the variables to expose in
// the sandbox for brokered credentials: a placeholder for each present
// credential, passed through the forwarding function.
func ResolveSandboxCredentialEnvironment(ctx context.Context, options SandboxCredentialEnvironmentOptions) (map[string]string, error) {
	resolved := make(map[string]string)

	for _, name := range uniqueNames(options.CredentialEnvironmentVariables) {
		if _, present := options.Environment[name]; !present {
			continue
		}

		if previous, ok := options.Previous[name]; ok {
			resolved[name] = previous

			continue
		}

		placeholder := NewSandboxCredentialPlaceholder()
		if options.Forwarding == nil {
			resolved[name] = placeholder

			continue
		}

		value, err := options.Forwarding(ctx, CredentialForwardingEvent{
			Credential:              placeholder,
			EnvironmentVariableName: name,
		})
		if err != nil {
			return nil, fmt.Errorf("forward credential placeholder %s: %w", name, err)
		}

		resolved[name] = value
	}

	return resolved, nil
}

// CredentialBrokeringCheck describes the credentials forwarded into a sandbox
// that cannot broker them.
type CredentialBrokeringCheck struct {
	// Environment holds the real credentials.
	Environment map[string]string

	// ForwardedEnvironment is what the sandbox receives.
	ForwardedEnvironment map[string]string

	CredentialEnvironmentVariables []string
}

// WarnCredentialBrokeringUnavailable logs a warning when real credentials
// are forwarded into a sandbox because it cannot install request
// transformations. It reports whether it warned.
func WarnCredentialBrokeringUnavailable(ctx context.Context, check CredentialBrokeringCheck) bool {
	names := uniqueNames(check.CredentialEnvironmentVariables)

	for _, name := range names {
		credential := check.Environment[name]
		if credential == "" {
			continue
		}

		for _, forwardedName := range names {
			if strings.Contains(check.ForwardedEnvironment[forwardedName], credential) {
				slog.Default().WarnContext(
					ctx,
					"sandbox cannot configure request transformations, so credential brokering is unavailable; falling back to less secure credential forwarding",
				)

				return true
			}
		}
	}

	return false
}

// MaskSandboxCredentials returns a copy of the environment with each
// credential replaced by its variable name, for logging.
func MaskSandboxCredentials(environment map[string]string, credentialEnvironmentVariables []string) map[string]string {
	masked := maps.Clone(environment)

	for _, name := range credentialEnvironmentVariables {
		if _, present := masked[name]; present {
			masked[name] = name
		}
	}

	return masked
}

// CredentialRequestTransformationOptions configure
// NewCredentialRequestTransformation.
type CredentialRequestTransformationOptions struct {
	// MatchURL selects the host and path prefix of the requests.
	MatchURL string

	// MatchHeaders are the exact headers the requests carry, typically the
	// placeholder.
	MatchHeaders map[string]string

	// TransformHeaders are the real credential headers to inject.
	TransformHeaders map[string]string
}

// NewCredentialRequestTransformation returns a request transformation that
// replaces placeholder headers with real credentials for one endpoint.
func NewCredentialRequestTransformation(options CredentialRequestTransformationOptions) (RequestTransformation, error) {
	parsed, err := url.Parse(options.MatchURL)
	if err != nil {
		return RequestTransformation{}, fmt.Errorf("parse credential url: %w", err)
	}

	match := RequestMatch{
		Host: parsed.Hostname(),
	}

	if pathname := strings.TrimRight(parsed.EscapedPath(), "/"); pathname != "" {
		match.Path = &Matcher{
			StartsWith: pathname,
		}
	}

	for _, key := range slices.Sorted(maps.Keys(options.MatchHeaders)) {
		match.Headers = append(match.Headers, KeyValueMatcher{
			Key: &Matcher{
				Exact: key,
			},
			Value: &Matcher{
				Exact: options.MatchHeaders[key],
			},
		})
	}

	return RequestTransformation{
		Match: match,
		Transform: RequestTransform{
			Headers: maps.Clone(options.TransformHeaders),
		},
	}, nil
}

// uniqueNames removes repeated names, keeping the first occurrence.
func uniqueNames(names []string) []string {
	unique := make([]string, 0, len(names))
	for _, name := range names {
		if !slices.Contains(unique, name) {
			unique = append(unique, name)
		}
	}

	return unique
}
