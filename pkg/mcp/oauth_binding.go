package mcp

import (
	"context"
	"net/url"
)

// newServerInformation returns the binding of the authorization server at
// authorizationServerURL: its issuer, normalized URL and token endpoint.
func newServerInformation(authorizationServerURL string, metadata *AuthorizationServerMetadata) (OAuthAuthorizationServerInformation, error) {
	serverURL, err := normalizeURL(authorizationServerURL)
	if err != nil {
		return OAuthAuthorizationServerInformation{}, err
	}

	issuer := authorizationServerURL
	if metadata != nil {
		issuer = metadata.Issuer
	}

	parsed, err := url.Parse(serverURL)
	if err != nil {
		return OAuthAuthorizationServerInformation{}, err
	}

	tokenURL, err := tokenEndpoint(parsed, metadata)
	if err != nil {
		return OAuthAuthorizationServerInformation{}, err
	}

	normalizedTokenEndpoint, err := normalizeURL(tokenURL.String())
	if err != nil {
		return OAuthAuthorizationServerInformation{}, err
	}

	return OAuthAuthorizationServerInformation{
		Issuer:                 issuer,
		AuthorizationServerURL: serverURL,
		TokenEndpoint:          normalizedTokenEndpoint,
	}, nil
}

// serverInformationFromCredentials returns the binding stored with tokens or
// client information, or nil when it is incomplete or invalid.
func serverInformationFromCredentials(issuer string, authorizationServer string, tokenEndpointURL string) *OAuthAuthorizationServerInformation {
	if authorizationServer == "" || tokenEndpointURL == "" {
		return nil
	}

	return normalizedServerInformation(OAuthAuthorizationServerInformation{
		Issuer:                 issuer,
		AuthorizationServerURL: authorizationServer,
		TokenEndpoint:          tokenEndpointURL,
	})
}

func normalizedServerInformation(information OAuthAuthorizationServerInformation) *OAuthAuthorizationServerInformation {
	serverURL, err := normalizeURL(information.AuthorizationServerURL)
	if err != nil {
		return nil
	}

	normalizedTokenEndpoint, err := normalizeURL(information.TokenEndpoint)
	if err != nil {
		return nil
	}

	return &OAuthAuthorizationServerInformation{
		Issuer:                 information.Issuer,
		AuthorizationServerURL: serverURL,
		TokenEndpoint:          normalizedTokenEndpoint,
	}
}

// storedServerInformation returns the stored binding from the tokens, the
// provider, or the client information, in that order.
func storedServerInformation(
	ctx context.Context,
	provider OAuthClientProvider,
	clientInformation OAuthClientInformation,
	tokens *OAuthTokens,
) (*OAuthAuthorizationServerInformation, error) {
	if tokens != nil {
		fromTokens := serverInformationFromCredentials(tokens.Issuer, tokens.AuthorizationServer, tokens.TokenEndpoint)
		if fromTokens != nil {
			return fromTokens, nil
		}
	}

	if loader, ok := provider.(OAuthAuthorizationServerInformationLoader); ok {
		information, err := loader.AuthorizationServerInformation(ctx)
		if err != nil {
			return nil, err
		}

		if information != nil {
			if normalized := normalizedServerInformation(*information); normalized != nil {
				return normalized, nil
			}
		}
	}

	return serverInformationFromCredentials(
		clientInformation.Issuer,
		clientInformation.AuthorizationServer,
		clientInformation.TokenEndpoint,
	), nil
}

// saveServerInformation stores the binding with the provider, or with the
// client information. It reports false when the provider can store neither.
func saveServerInformation(
	ctx context.Context,
	provider OAuthClientProvider,
	clientInformation OAuthClientInformation,
	information OAuthAuthorizationServerInformation,
) (bool, error) {
	if saver, ok := provider.(OAuthAuthorizationServerInformationSaver); ok {
		return true, saver.SaveAuthorizationServerInformation(ctx, information)
	}

	if saver, ok := provider.(OAuthClientInformationSaver); ok {
		return true, saver.SaveClientInformation(ctx, bindClientInformation(clientInformation, information))
	}

	return false, nil
}

// assertServerInformationMatches requires the current authorization server
// to be the one that issued the stored credentials.
func assertServerInformationMatches(stored OAuthAuthorizationServerInformation, current OAuthAuthorizationServerInformation) error {
	issuerDiffers := stored.Issuer != "" && current.Issuer != "" && stored.Issuer != current.Issuer

	if issuerDiffers ||
		stored.AuthorizationServerURL != current.AuthorizationServerURL ||
		stored.TokenEndpoint != current.TokenEndpoint {
		return authorizationServerMismatch(
			"oauth authorization server metadata does not match the metadata that issued the stored credentials",
		)
	}

	return nil
}

func bindTokens(tokens OAuthTokens, information OAuthAuthorizationServerInformation) OAuthTokens {
	tokens.Issuer = information.Issuer
	tokens.AuthorizationServer = information.AuthorizationServerURL
	tokens.TokenEndpoint = information.TokenEndpoint

	return tokens
}

func bindClientInformation(clientInformation OAuthClientInformation, information OAuthAuthorizationServerInformation) OAuthClientInformation {
	clientInformation.Issuer = information.Issuer
	clientInformation.AuthorizationServer = information.AuthorizationServerURL
	clientInformation.TokenEndpoint = information.TokenEndpoint

	return clientInformation
}
