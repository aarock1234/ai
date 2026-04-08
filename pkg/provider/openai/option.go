package openai

import "net/http"

const defaultBaseURL = "https://api.openai.com/v1"

// config holds OpenAI provider settings.
type config struct {
	apiKey       string
	baseURL      string
	organization string
	httpClient   *http.Client
}

// Option configures the OpenAI provider.
type Option func(*config)

// WithAPIKey sets the API key. If not set, the provider reads
// OPENAI_API_KEY from the environment.
func WithAPIKey(key string) Option {
	return func(c *config) { c.apiKey = key }
}

// WithBaseURL overrides the API base URL. Useful for proxies or
// Azure OpenAI endpoints.
func WithBaseURL(url string) Option {
	return func(c *config) { c.baseURL = url }
}

// WithOrganization sets the OpenAI-Organization header.
func WithOrganization(org string) Option {
	return func(c *config) { c.organization = org }
}

// WithHTTPClient sets a custom HTTP client for all requests.
func WithHTTPClient(client *http.Client) Option {
	return func(c *config) { c.httpClient = client }
}
