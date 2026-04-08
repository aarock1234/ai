package openrouter

import "net/http"

// config holds OpenRouter provider settings.
type config struct {
	apiKey      string
	baseURL     string
	httpReferer string
	siteURL     string
	httpClient  *http.Client
}

// Option configures the OpenRouter provider.
type Option func(*config)

// WithAPIKey sets the API key. If not set, the provider reads
// OPENROUTER_API_KEY from the environment.
func WithAPIKey(key string) Option {
	return func(c *config) { c.apiKey = key }
}

// WithBaseURL overrides the API base URL.
func WithBaseURL(url string) Option {
	return func(c *config) { c.baseURL = url }
}

// WithHTTPReferer sets the HTTP-Referer header sent with each request.
// OpenRouter uses this to identify the originating site.
func WithHTTPReferer(referer string) Option {
	return func(c *config) { c.httpReferer = referer }
}

// WithSiteURL sets the X-Title header sent with each request.
// OpenRouter uses this for display purposes in the dashboard.
func WithSiteURL(url string) Option {
	return func(c *config) { c.siteURL = url }
}

// WithHTTPClient sets a custom HTTP client for all requests.
func WithHTTPClient(client *http.Client) Option {
	return func(c *config) { c.httpClient = client }
}
