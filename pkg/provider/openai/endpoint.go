package openai

import (
	"context"
	"net/http"
)

// Endpoint identifies the target of one request.
type Endpoint struct {
	// Path is the API path, such as "/responses".
	Path string

	// ModelID is the model ID sent in the request, which hosts such as Azure
	// OpenAI use as a deployment name.
	ModelID string
}

// RequestURLFunc returns the full URL of one request. Its error is returned
// from the call that needed the URL.
type RequestURLFunc func(endpoint Endpoint) (string, error)

// WithRequestURL replaces joining the base URL and the request path, for
// hosts that route by deployment or require query parameters.
func WithRequestURL(requestURL RequestURLFunc) Option {
	return func(c *config) {
		c.requestURL = requestURL
	}
}

// AuthHeadersFunc returns the authentication headers of one request. Its
// headers may be empty when authentication happens elsewhere, such as in
// the HTTP client's transport. Its error is returned from the call.
type AuthHeadersFunc func(ctx context.Context) (http.Header, error)

// WithAuthHeaders replaces the bearer Authorization header built from the
// API key. authHeaders is called for every request.
func WithAuthHeaders(authHeaders AuthHeadersFunc) Option {
	return func(c *config) {
		c.authHeaders = authHeaders
	}
}

// WithOptionsNamespace sets the provider options namespace that Responses
// API models read before Namespace and that keys their metadata, such as
// "azure" for Azure OpenAI. The default is Namespace.
func WithOptionsNamespace(namespace string) Option {
	return func(c *config) {
		c.optionsNamespace = namespace
	}
}

// WithExplicitMessageItems sends "type": "message" with Responses API message
// items, as Azure AI Foundry project endpoints require.
func WithExplicitMessageItems() Option {
	return func(c *config) {
		c.explicitMessageItems = true
	}
}

// WithoutWebSearchSourcesInclude stops Responses API calls with a web search
// tool from adding "web_search_call.action.sources" to the include list, for
// hosts such as Amazon Bedrock Mantle that reject it.
func WithoutWebSearchSourcesInclude() Option {
	return func(c *config) {
		c.omitWebSearchSourcesInclude = true
	}
}

// ModelOption configures one language model created by a Provider.
type ModelOption func(*config)

// WithRequestModelID sends requestModelID as the model of each request and
// passes it to the URL builder, while ModelID and capability detection keep
// the model ID. Hosts that address user-named deployments, such as Azure
// OpenAI, use it.
func WithRequestModelID(requestModelID string) ModelOption {
	return func(c *config) {
		c.requestModelID = requestModelID
	}
}

// forModel returns the configuration of one model.
func (c config) forModel(modelID string, options []ModelOption) config {
	c.requestModelID = modelID

	for _, option := range options {
		option(&c)
	}

	return c
}
