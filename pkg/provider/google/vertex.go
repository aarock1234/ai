package google

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

const (
	// vertexNamespace and legacyVertexNamespace key the options and metadata
	// of Vertex AI models.
	vertexNamespace       = "googleVertex"
	legacyVertexNamespace = "vertex"

	// defaultToolResultDownloadMaxBytes bounds each tool result file that a
	// Vertex AI model downloads: 7 MiB.
	defaultToolResultDownloadMaxBytes int64 = 7 << 20
)

// errVertexFileReference reports a provider file reference, which Vertex AI
// cannot resolve.
var errVertexFileReference = fmt.Errorf("%w: file parts with provider references", model.ErrUnsupported)

var (
	httpURLPattern               = regexp.MustCompile(`^https?://.*$`)
	googleCloudStorageURLPattern = regexp.MustCompile(`^gs://.*$`)
)

// googleCloudStorageFunctionResponseMediaTypes are the media types that
// Gemini 3 models on Vertex AI read from Google Cloud Storage URLs inside
// function responses.
var googleCloudStorageFunctionResponseMediaTypes = []string{
	"image/png",
	"image/jpeg",
	"image/webp",
	"application/pdf",
	"text/plain",
}

// VertexAISettings configure Vertex AI behavior.
type VertexAISettings struct {
	// ToolResultDownloadMaxBytes bounds each tool result file URL that is
	// downloaded and sent inline. Zero selects 7 MiB.
	ToolResultDownloadMaxBytes int64
}

// WithVertexAI makes models speak the Vertex AI dialect of the
// generateContent API: provider options are read from the "googleVertex"
// and "vertex" namespaces, falling back to "google", and metadata is written
// under both; Vertex-only options such as request types, streamed function
// call arguments and image person generation are sent; HTTP(S) and gs:// URLs
// are passed to the model; tool result file URLs are downloaded; and function
// call IDs are omitted. Combine it with WithBaseURLResolver and
// WithAuthHeaders or WithAPIKey to reach a Vertex AI endpoint.
func WithVertexAI(settings VertexAISettings) Option {
	return func(p *Provider) {
		if settings.ToolResultDownloadMaxBytes <= 0 {
			settings.ToolResultDownloadMaxBytes = defaultToolResultDownloadMaxBytes
		}

		p.vertex = &settings
	}
}

// isVertex reports whether the provider uses the Vertex AI dialect.
func (p *Provider) isVertex() bool {
	return p.vertex != nil
}

// optionNamespaces lists the namespaces of model options in read order.
func (p *Provider) optionNamespaces() []string {
	if p.isVertex() {
		return []string{
			vertexNamespace,
			legacyVertexNamespace,
			Namespace,
		}
	}

	return []string{
		Namespace,
	}
}

// partMetadataNamespaces lists the namespaces of replayed part metadata in
// read order. Each dialect accepts the other's namespaces for gateway
// interoperability.
func (p *Provider) partMetadataNamespaces() []string {
	if p.isVertex() {
		return []string{
			vertexNamespace,
			legacyVertexNamespace,
			Namespace,
		}
	}

	return []string{
		Namespace,
		vertexNamespace,
		legacyVertexNamespace,
	}
}

// metadataNamespaces lists the namespaces that output metadata is written
// under.
func (p *Provider) metadataNamespaces() []string {
	if p.isVertex() {
		return []string{
			vertexNamespace,
			legacyVertexNamespace,
		}
	}

	return []string{
		Namespace,
	}
}

// wrapMetadata stores value under every namespace.
func wrapMetadata(namespaces []string, value any) model.ProviderMetadata {
	metadata := make(model.ProviderMetadata, len(namespaces))
	for _, namespace := range namespaces {
		metadata[namespace] = value
	}

	return metadata
}

// readProviderValue returns the value of the first namespace that holds one.
func readProviderValue[T any](options model.ProviderOptions, namespaces []string) (T, bool, error) {
	for _, namespace := range namespaces {
		value, ok, err := model.ProviderValue[T](options, namespace)
		if err != nil || ok {
			return value, ok, err
		}
	}

	var zero T

	return zero, false, nil
}

// vertexSupportedURLs are the URLs Vertex AI models download themselves:
// HTTP(S) and Google Cloud Storage URLs of every media type.
func vertexSupportedURLs() map[string][]*regexp.Regexp {
	return map[string][]*regexp.Regexp{
		"*": {
			httpURLPattern,
			googleCloudStorageURLPattern,
		},
	}
}

// isURLSupported reports whether a pattern of the media type, of a matching
// wildcard such as "image/*", or of "*" matches the URL. Both are compared in
// lowercase.
func isURLSupported(location string, mediaType string, supported map[string][]*regexp.Regexp) bool {
	location = strings.ToLower(location)
	mediaType = strings.ToLower(mediaType)

	for key, patterns := range supported {
		prefix := strings.ToLower(key)
		if prefix == "*" || prefix == "*/*" {
			prefix = ""
		}

		prefix = strings.Replace(prefix, "*", "", 1)
		if !strings.HasPrefix(mediaType, prefix) {
			continue
		}

		for _, pattern := range patterns {
			if pattern.MatchString(location) {
				return true
			}
		}
	}

	return false
}

// functionResponseURLs are the file URLs that function responses may
// reference: Google Cloud Storage URLs on Vertex AI with Gemini 3 models,
// otherwise none.
func (p *Provider) functionResponseURLs(isGemini3 bool) map[string][]*regexp.Regexp {
	if !p.isVertex() || !isGemini3 {
		return nil
	}

	supported := make(map[string][]*regexp.Regexp, len(googleCloudStorageFunctionResponseMediaTypes))
	for _, mediaType := range googleCloudStorageFunctionResponseMediaTypes {
		supported[mediaType] = []*regexp.Regexp{
			googleCloudStorageURLPattern,
		}
	}

	return supported
}
