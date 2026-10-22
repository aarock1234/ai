package googlevertex

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/apikey"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/googlevertex/internal/vertex"
)

// expressModeBaseURL is the URL prefix of Vertex AI express mode.
const expressModeBaseURL = "https://aiplatform.googleapis.com/v1/publishers/google"

// endpointModelPrefix starts the IDs of tuned models, which are served from
// a deployed endpoint addressed as endpoints/{id} instead of
// publishers/google/models/{id}.
const endpointModelPrefix = "endpoints/"

func isEndpointModel(modelID string) bool {
	return strings.HasPrefix(modelID, endpointModelPrefix)
}

// loadAPIKey returns the express mode API key, or an empty string when
// requests use the token source.
func (s *settings) loadAPIKey() string {
	return apikey.Lookup(s.apiKey, apiKeyVariable)
}

// isExpressMode reports whether requests authenticate with an API key.
func (s *settings) isExpressMode() bool {
	return s.loadAPIKey() != ""
}

// authHeaders returns the API key header in express mode, or a bearer token
// from the token source.
func (s *settings) authHeaders(ctx context.Context) (http.Header, error) {
	if apiKey := s.loadAPIKey(); apiKey != "" {
		return http.Header{
			"X-Goog-Api-Key": {
				apiKey,
			},
		}, nil
	}

	if s.tokenSource == nil {
		return nil, fmt.Errorf(
			"%w: set %s, use WithAPIKey for express mode, or use WithTokenSource",
			model.ErrLoadAPIKey,
			apiKeyVariable,
		)
	}

	token, err := s.tokenSource(ctx)
	if err != nil {
		return nil, fmt.Errorf("google vertex token source: %w", err)
	}

	return http.Header{
		"Authorization": {
			"Bearer " + token,
		},
	}, nil
}

// requireTokenSource rejects express mode for models that need standard
// Google Cloud credentials.
func (s *settings) requireTokenSource(models string) error {
	if !s.isExpressMode() {
		return nil
	}

	return fmt.Errorf(
		"%w: Google Vertex %s do not support Express Mode API keys; use standard Google Cloud credentials instead",
		model.ErrUnsupported,
		models,
	)
}

// resolveBaseURL returns the URL prefix of Vertex AI requests. Tuned model
// endpoints omit the /publishers/google suffix and need a token source.
func (s *settings) resolveBaseURL(endpoint bool) (string, error) {
	if endpoint {
		if err := s.requireTokenSource("tuned models"); err != nil {
			return "", err
		}
	}

	if s.isExpressMode() {
		if s.baseURL != "" {
			return s.baseURL, nil
		}

		return expressModeBaseURL, nil
	}

	if s.baseURL != "" {
		return s.baseURL, nil
	}

	return s.regionalBaseURL(endpoint)
}

// regionalBaseURL returns the project and location URL prefix.
func (s *settings) regionalBaseURL(endpoint bool) (string, error) {
	location, err := vertex.LoadLocation(s.location, "")
	if err != nil {
		return "", err
	}

	project, err := vertex.LoadProject(s.project)
	if err != nil {
		return "", err
	}

	baseURL := "https://" + vertex.Host(location) + "/v1beta1/projects/" + project + "/locations/" + location
	if !endpoint {
		baseURL += "/publishers/google"
	}

	return baseURL, nil
}

// request describes a call to requestURL with the authentication headers,
// the provider headers and the call headers.
func (s *settings) request(ctx context.Context, requestURL string, callHeaders http.Header) (apicall.Request, error) {
	authentication, err := s.authHeaders(ctx)
	if err != nil {
		return apicall.Request{}, err
	}

	return apicall.Request{
		Client:     s.httpClient,
		URL:        requestURL,
		Headers:    apicall.MergeHeaders(authentication, s.headers, callHeaders),
		ParseError: parseError,
	}, nil
}
