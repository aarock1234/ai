package mantle

import (
	"strings"

	"github.com/aarock1234/ai/internal/awssigv4"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

// openAIRouteModelPrefixes are the model ID prefixes that Mantle serves under
// its separate /openai/v1 route.
var openAIRouteModelPrefixes = []string{
	"openai.gpt-",
	"google.gemma-4",
	"xai.",
}

// requestURL returns the URL of one request: the configured base URL, or the
// regional Mantle endpoint of the request's model, followed by the path.
func (p *Provider) requestURL(endpoint openai.Endpoint) (string, error) {
	baseURL, err := p.baseURL(endpoint.ModelID)
	if err != nil {
		return "", err
	}

	return baseURL + endpoint.Path, nil
}

// baseURL returns the configured base URL without a trailing slash, or the
// regional endpoint and route of modelID.
func (p *Provider) baseURL(modelID string) (string, error) {
	if p.config.BaseURL != "" {
		return strings.TrimSuffix(p.config.BaseURL, "/"), nil
	}

	region, err := awssigv4.LoadHostRegion(p.config.Region)
	if err != nil {
		return "", err
	}

	return "https://bedrock-mantle." + region + ".api.aws/" + routeOf(modelID), nil
}

// routeOf returns the API route of a model: "openai/v1" for GPT models other
// than gpt-oss, Gemma 4 and xAI models, and "v1" otherwise.
func routeOf(modelID string) string {
	if strings.HasPrefix(modelID, "openai.gpt-oss-") {
		return "v1"
	}

	for _, prefix := range openAIRouteModelPrefixes {
		if strings.HasPrefix(modelID, prefix) {
			return "openai/v1"
		}
	}

	return "v1"
}

// serviceBaseURL returns the base URL whose origin receives credentials.
// Every route of the regional endpoint shares one origin.
func (p *Provider) serviceBaseURL() (string, error) {
	return p.baseURL("")
}
