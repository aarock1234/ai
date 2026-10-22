package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"net/http"
	"regexp"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// hostSettings adapt Messages API requests for platforms that host Claude,
// such as Amazon Bedrock, Google Vertex AI and Claude Platform on AWS. The
// zero value sends requests to the Anthropic API unchanged.
type hostSettings struct {
	headers          AuthHeadersFunc
	baseURL          func() (string, error)
	requestURL       func(stream bool) (string, error)
	transformBody    func(body json.RawMessage, betas []string) (json.RawMessage, error)
	transformHeaders func(headers http.Header, options model.ProviderOptions) (http.Header, error)
	supportedURLs    map[string][]*regexp.Regexp

	disableNativeStructuredOutput bool
	disableStrictTools            bool

	convertMediaPart         MediaPartConverter
	maxTemperature           *float64
	keepSamplingWithThinking bool
}

// MediaContentType is the type of a media content block built by a
// MediaPartConverter.
type MediaContentType string

// Media content types.
const (
	MediaContentImage MediaContentType = "image"
	MediaContentVideo MediaContentType = "video"
)

// MediaContent is an image or video content block built for an
// Anthropic-compatible API. The block's cache control is added by the
// provider.
type MediaContent struct {
	Type MediaContentType

	// Source is the encoded source object of the block, such as a base64 or
	// url source with members specific to the compatible API.
	Source json.RawMessage
}

// MediaPartConverter converts a file part of a user message into a media
// content block. It returns nil to use the default conversion.
type MediaPartConverter func(part model.FilePart) (*MediaContent, error)

// WithMediaPartConverter sets the function that converts file parts of user
// messages for an Anthropic-compatible API before the default conversion.
func WithMediaPartConverter(convert MediaPartConverter) Option {
	return func(p *Provider) {
		p.host.convertMediaPart = convert
	}
}

// WithMaxTemperature replaces the maximum temperature of 1 for an
// Anthropic-compatible API. Higher temperatures are clamped with a warning
// that names the provider's options namespace.
func WithMaxTemperature(maximum float64) Option {
	return func(p *Provider) {
		p.host.maxTemperature = &maximum
	}
}

// WithTemperatureAndTopPWithThinking sets whether an Anthropic-compatible
// API accepts temperature and topP while thinking is enabled. The default is
// false, which drops them with a warning.
func WithTemperatureAndTopPWithThinking(supported bool) Option {
	return func(p *Provider) {
		p.host.keepSamplingWithThinking = supported
	}
}

// AuthHeadersFunc returns the authentication headers of one request. Its
// headers may be empty when authentication happens elsewhere, such as in
// the HTTP client's transport. Its error is returned from the call.
type AuthHeadersFunc func(ctx context.Context) (http.Header, error)

// WithAuthHeaders sets the function that returns the version and
// authentication headers of each request, replacing the anthropic-version,
// x-api-key and Authorization headers the provider would send. Headers set
// with WithHeaders are added after them. The API key and auth token options
// are then unused.
func WithAuthHeaders(authHeaders AuthHeadersFunc) Option {
	return func(p *Provider) {
		p.host.headers = authHeaders
	}
}

// WithBaseURLResolver sets the function that returns the base URL of each
// request when it is sent, such as a regional endpoint, replacing
// WithBaseURL and ANTHROPIC_BASE_URL. The URL must not end with a slash.
func WithBaseURLResolver(resolve func() (string, error)) Option {
	return func(p *Provider) {
		p.host.baseURL = resolve
	}
}

// WithRequestURL sets the function that returns the URL of each call,
// replacing the base URL and the /messages path. stream reports whether the
// call streams.
func WithRequestURL(requestURL func(stream bool) (string, error)) Option {
	return func(p *Provider) {
		p.host.requestURL = requestURL
	}
}

// WithRequestBodyTransform sets a function that rewrites the JSON request
// body before it is sent. betas are the beta features the request uses, in
// order; they are also sent in the anthropic-beta header.
func WithRequestBodyTransform(transform func(body json.RawMessage, betas []string) (json.RawMessage, error)) Option {
	return func(p *Provider) {
		p.host.transformBody = transform
	}
}

// WithRequestHeadersTransform sets a function that rewrites the headers of
// each request, after all other headers are set. options are the call's
// provider options.
func WithRequestHeadersTransform(transform func(headers http.Header, options model.ProviderOptions) (http.Header, error)) Option {
	return func(p *Provider) {
		p.host.transformHeaders = transform
	}
}

// WithSupportedURLs sets the URL patterns, by media type, that the models
// report as downloaded by the API itself. An empty map makes every file URL
// download locally. The map is copied.
func WithSupportedURLs(urls map[string][]*regexp.Regexp) Option {
	snapshot := make(map[string][]*regexp.Regexp, len(urls))
	for mediaType, patterns := range urls {
		snapshot[mediaType] = slices.Clone(patterns)
	}

	return func(p *Provider) {
		p.host.supportedURLs = maps.Clone(snapshot)
	}
}

// WithNativeStructuredOutput sets whether the host accepts
// output_config.format. When false, JSON responses with a schema use the
// JSON response tool. The default is true.
func WithNativeStructuredOutput(supported bool) Option {
	return func(p *Provider) {
		p.host.disableNativeStructuredOutput = !supported
	}
}

// WithStrictTools sets whether the host accepts strict function tools. When
// false, strict settings are dropped with a warning. The default is true.
func WithStrictTools(supported bool) Option {
	return func(p *Provider) {
		p.host.disableStrictTools = !supported
	}
}

// requestURL returns the URL of a call: the host's URL, or the base URL
// with the /messages path.
func (p *Provider) requestURL(stream bool) (string, error) {
	if p.host.requestURL != nil {
		return p.host.requestURL(stream)
	}

	baseURL, err := p.resolveBaseURL()
	if err != nil {
		return "", err
	}

	return baseURL + "/messages", nil
}

// restrict removes the capabilities the host does not offer.
func (h hostSettings) restrict(capabilities modelCapabilities) modelCapabilities {
	if h.disableNativeStructuredOutput {
		capabilities.supportsStructuredOutput = false
	}

	if h.disableStrictTools {
		capabilities.supportsStrictTools = false
	}

	return capabilities
}

func (h hostSettings) transformedHeaders(headers http.Header, options model.ProviderOptions) (http.Header, error) {
	if h.transformHeaders == nil {
		return headers, nil
	}

	return h.transformHeaders(headers, options)
}

// transformedBody returns the body to send: the request itself, or its JSON
// after the host's transform.
func (h hostSettings) transformedBody(body messagesRequest, betas []string) (any, error) {
	if h.transformBody == nil {
		return body, nil
	}

	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("encode request body: %w", err)
	}

	return h.transformBody(encoded, slices.Clone(betas))
}
