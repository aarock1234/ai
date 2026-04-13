package cohere

import "net/http"

const defaultBaseURL = "https://api.cohere.com/v2"

type config struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

// Option configures the Cohere provider.
type Option func(*config)

// WithAPIKey sets the API key. If not set, the provider reads
// COHERE_API_KEY from the environment.
func WithAPIKey(key string) Option {
	return func(c *config) { c.apiKey = key }
}

// WithBaseURL overrides the API base URL.
func WithBaseURL(url string) Option {
	return func(c *config) { c.baseURL = url }
}

// WithHTTPClient sets a custom HTTP client for all requests.
func WithHTTPClient(client *http.Client) Option {
	return func(c *config) { c.httpClient = client }
}
