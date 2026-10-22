package bedrock

import (
	"context"

	"github.com/aarock1234/ai/internal/awssigv4"
)

// Credentials are AWS credentials that sign requests with SigV4.
type Credentials struct {
	AccessKeyID     string
	SecretAccessKey string

	// SessionToken is set for temporary credentials.
	SessionToken string
}

// CredentialProvider returns credentials for each request, such as
// temporary credentials that rotate.
type CredentialProvider func(ctx context.Context) (Credentials, error)

// SigningCredentials adapts a public credential provider to the signer. A
// nil provider stays nil.
func SigningCredentials(provider CredentialProvider) awssigv4.CredentialProvider {
	if provider == nil {
		return nil
	}

	return func(ctx context.Context) (awssigv4.Credentials, error) {
		credentials, err := provider(ctx)
		if err != nil {
			return awssigv4.Credentials{}, err
		}

		return awssigv4.Credentials{
			AccessKeyID:     credentials.AccessKeyID,
			SecretAccessKey: credentials.SecretAccessKey,
			SessionToken:    credentials.SessionToken,
		}, nil
	}
}
