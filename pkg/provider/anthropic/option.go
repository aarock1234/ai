package anthropic

import "net/http"

const (
	defaultBaseURL   = "https://api.anthropic.com/v1"
	defaultVersion   = "2023-06-01"
	defaultMaxTokens = 1024
)

// config holds Anthropic provider settings.
type config struct {
	apiKey     string
	baseURL    string
	apiVersion string
	httpClient *http.Client
}

// Option configures the Anthropic provider.
type Option func(*config)

// WithAPIKey sets the API key. If not set, the provider reads
// ANTHROPIC_API_KEY from the environment.
func WithAPIKey(key string) Option {
	return func(c *config) { c.apiKey = key }
}

// WithBaseURL overrides the API base URL.
func WithBaseURL(url string) Option {
	return func(c *config) { c.baseURL = url }
}

// WithAPIVersion sets the anthropic-version header sent with each request.
func WithAPIVersion(version string) Option {
	return func(c *config) { c.apiVersion = version }
}

// WithHTTPClient sets a custom HTTP client for all requests.
func WithHTTPClient(client *http.Client) Option {
	return func(c *config) { c.httpClient = client }
}
