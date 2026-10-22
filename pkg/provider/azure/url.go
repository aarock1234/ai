package azure

import (
	"fmt"
	"net/url"
	"os"
	"regexp"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

// hostnamePartPattern matches a single DNS label.
var hostnamePartPattern = regexp.MustCompile(`(?i)^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$`)

// baseURLInfo classifies a custom base URL.
type baseURLInfo struct {
	// isAzureOpenAI reports an Azure OpenAI, AI Services or Cognitive
	// Services host, as opposed to a custom gateway.
	isAzureOpenAI bool

	// isFoundryProject reports an Azure AI Foundry project endpoint.
	isFoundryProject bool

	// isVersioned reports an Azure OpenAI URL that already ends in
	// /openai/v1.
	isVersioned bool
}

// baseURLInfo classifies the configured base URL. Without one, requests go
// to the unversioned Azure OpenAI resource URL.
func (s settings) baseURLInfo() baseURLInfo {
	if s.baseURL == "" {
		return baseURLInfo{
			isAzureOpenAI: true,
		}
	}

	parsed, err := url.Parse(s.baseURL)
	if err != nil {
		return baseURLInfo{}
	}

	hostname := parsed.Hostname()
	isAzureOpenAI := strings.HasSuffix(hostname, ".openai.azure.com") ||
		strings.HasSuffix(hostname, ".services.ai.azure.com") ||
		strings.HasSuffix(hostname, ".cognitiveservices.azure.com")

	path := strings.TrimRight(parsed.Path, "/")

	return baseURLInfo{
		isAzureOpenAI:    isAzureOpenAI,
		isFoundryProject: strings.HasSuffix(hostname, ".services.ai.azure.com") && strings.HasPrefix(path, "/api/projects/"),
		isVersioned:      isAzureOpenAI && strings.HasSuffix(strings.ToLower(path), "/openai/v1"),
	}
}

// openaiURL builds the URL of an OpenAI API request. Deployment-based URLs
// address the deployment in the path; v1 URLs send it as the request model.
// Azure OpenAI URLs that do not carry their own version receive the
// api-version query parameter.
func (s settings) openaiURL(endpoint openai.Endpoint) (string, error) {
	prefix, err := s.openaiURLPrefix()
	if err != nil {
		return "", err
	}

	info := s.baseURLInfo()

	var location string
	switch {
	case s.useDeploymentBasedURLs:
		location = prefix + "/deployments/" + endpoint.ModelID + endpoint.Path
	case !info.isAzureOpenAI || info.isVersioned:
		location = prefix + endpoint.Path
	default:
		location = prefix + "/v1" + endpoint.Path
	}

	parsed, err := url.Parse(location)
	if err != nil {
		return "", fmt.Errorf("%w: azure request URL: %w", model.ErrInvalidArgument, err)
	}

	sendsVersion := info.isAzureOpenAI && !info.isVersioned && !info.isFoundryProject
	if s.useDeploymentBasedURLs || sendsVersion {
		query := parsed.Query()
		query.Set("api-version", s.apiVersion)
		parsed.RawQuery = query.Encode()
	}

	return parsed.String(), nil
}

// openaiURLPrefix returns the base URL, or the resource URL.
func (s settings) openaiURLPrefix() (string, error) {
	if s.baseURL != "" {
		return s.baseURL, nil
	}

	resourceName, err := s.loadResourceName()
	if err != nil {
		return "", err
	}

	return "https://" + resourceName + ".openai.azure.com/openai", nil
}

// maiURL returns the URL of a MAI API path, such as "/images/generations".
func (s settings) maiURL(path string) (string, error) {
	if s.maiBaseURL != "" {
		return s.maiBaseURL + path, nil
	}

	resourceName, err := s.loadResourceName()
	if err != nil {
		return "", err
	}

	return "https://" + resourceName + ".services.ai.azure.com/mai/v1" + path, nil
}

// speechURL returns the URL of an Azure Speech path, such as
// "/tts/cognitiveservices/v1".
func (s settings) speechURL(path string) (string, error) {
	if s.speechBaseURL != "" {
		return s.speechBaseURL + path, nil
	}

	resourceName, err := s.loadResourceName()
	if err != nil {
		return "", err
	}

	return "https://" + resourceName + ".cognitiveservices.azure.com" + path, nil
}

// loadResourceName returns the configured resource name or reads
// AZURE_RESOURCE_NAME. The name becomes part of the request host, so only a
// single DNS label is accepted.
func (s settings) loadResourceName() (string, error) {
	resourceName := s.resourceName
	if resourceName == "" {
		resourceName = os.Getenv(resourceNameVariable)
	}

	if resourceName == "" {
		return "", fmt.Errorf(
			"%w: Azure OpenAI resource name is missing; set %s or use WithResourceName",
			model.ErrInvalidArgument,
			resourceNameVariable,
		)
	}

	if !hostnamePartPattern.MatchString(resourceName) {
		return "", fmt.Errorf(
			"%w: invalid Azure resource name; expected a single DNS label (letters, digits, and hyphens); use WithBaseURL for custom endpoints",
			model.ErrInvalidArgument,
		)
	}

	return resourceName, nil
}
