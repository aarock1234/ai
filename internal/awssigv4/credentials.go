package awssigv4

import (
	"context"
	"fmt"
	"os"
	"regexp"

	"github.com/aarock1234/ai/pkg/model"
)

// Environment variables read for credentials and the region.
const (
	AccessKeyIDVariable     = "AWS_ACCESS_KEY_ID"
	SecretAccessKeyVariable = "AWS_SECRET_ACCESS_KEY"
	SessionTokenVariable    = "AWS_SESSION_TOKEN"
	RegionVariable          = "AWS_REGION"
)

// hostnameLabel matches a single DNS label, which a region must be to form
// an endpoint host.
var hostnameLabel = regexp.MustCompile(`^(?i)[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$`)

// CredentialProvider returns credentials on demand, such as temporary
// credentials that rotate.
type CredentialProvider func(ctx context.Context) (Credentials, error)

// CredentialSettings are the configured credential sources of a provider.
type CredentialSettings struct {
	// AccessKeyID, SecretAccessKey and SessionToken are explicit
	// credentials. Empty values fall back to the environment.
	AccessKeyID     string
	SecretAccessKey string
	SessionToken    string

	// Provider, when set, supplies the credentials instead of the explicit
	// values and the environment.
	Provider CredentialProvider

	// ExplicitKeysOwnSessionToken stops the session token from being read
	// from the environment when both keys are configured explicitly.
	ExplicitKeysOwnSessionToken bool

	// APIKeyAlternative names the API key alternative in the error for
	// missing credentials, such as "AWS_BEARER_TOKEN_BEDROCK or WithAPIKey".
	APIKeyAlternative string
}

// LoadCredentials resolves credentials from the provider, the explicit
// values or the environment. Missing keys produce an error wrapping
// model.ErrLoadAPIKey.
func LoadCredentials(ctx context.Context, settings CredentialSettings) (Credentials, error) {
	if settings.Provider != nil {
		return loadProvidedCredentials(ctx, settings.Provider)
	}

	accessKeyID := settingOrEnvironment(settings.AccessKeyID, AccessKeyIDVariable)
	if accessKeyID == "" {
		return Credentials{}, fmt.Errorf(
			"%w: AWS SigV4 authentication requires AWS credentials; set %s and %s, configure the access key ID and secret access key, use a credential provider, or use API key authentication with %s",
			model.ErrLoadAPIKey,
			AccessKeyIDVariable,
			SecretAccessKeyVariable,
			settings.APIKeyAlternative,
		)
	}

	secretAccessKey := settingOrEnvironment(settings.SecretAccessKey, SecretAccessKeyVariable)
	if secretAccessKey == "" {
		return Credentials{}, fmt.Errorf(
			"%w: AWS SigV4 authentication requires both %s and %s",
			model.ErrLoadAPIKey,
			AccessKeyIDVariable,
			SecretAccessKeyVariable,
		)
	}

	sessionToken := settingOrEnvironment(settings.SessionToken, SessionTokenVariable)

	explicitKeys := settings.AccessKeyID != "" && settings.SecretAccessKey != ""
	if settings.ExplicitKeysOwnSessionToken && explicitKeys {
		sessionToken = settings.SessionToken
	}

	return Credentials{
		AccessKeyID:     accessKeyID,
		SecretAccessKey: secretAccessKey,
		SessionToken:    sessionToken,
	}, nil
}

func loadProvidedCredentials(ctx context.Context, provider CredentialProvider) (Credentials, error) {
	credentials, err := provider(ctx)
	if err != nil {
		return Credentials{}, fmt.Errorf(
			"%w: AWS credential provider failed; ensure it returns valid AWS credentials with an access key ID and secret access key: %w",
			model.ErrLoadAPIKey,
			err,
		)
	}

	if credentials.AccessKeyID == "" || credentials.SecretAccessKey == "" {
		return Credentials{}, fmt.Errorf(
			"%w: AWS credential provider returned credentials without an access key ID or secret access key",
			model.ErrLoadAPIKey,
		)
	}

	return credentials, nil
}

// LoadRegion returns the configured region or AWS_REGION. A missing region
// produces an error wrapping model.ErrInvalidArgument.
func LoadRegion(configured string) (string, error) {
	region := settingOrEnvironment(configured, RegionVariable)
	if region == "" {
		return "", fmt.Errorf(
			"%w: AWS region setting is missing; configure the region or set the %s environment variable",
			model.ErrInvalidArgument,
			RegionVariable,
		)
	}

	return region, nil
}

// LoadHostRegion returns the region like LoadRegion and verifies that it is
// a single DNS label, so that it can form an endpoint host.
func LoadHostRegion(configured string) (string, error) {
	region, err := LoadRegion(configured)
	if err != nil {
		return "", err
	}

	if !IsHostnameLabel(region) {
		return "", fmt.Errorf(
			"%w: invalid AWS region %q; expected a single DNS label (letters, digits, and hyphens); use a base URL for custom endpoints",
			model.ErrInvalidArgument,
			region,
		)
	}

	return region, nil
}

// IsHostnameLabel reports whether value is a single DNS label of letters,
// digits and inner hyphens.
func IsHostnameLabel(value string) bool {
	return hostnameLabel.MatchString(value)
}

func settingOrEnvironment(setting string, variable string) string {
	if setting != "" {
		return setting
	}

	return os.Getenv(variable)
}
