package xai

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// Provider tool IDs.
const (
	toolWebSearch       = "xai.web_search"
	toolXSearch         = "xai.x_search"
	toolCodeExecution   = "xai.code_execution"
	toolViewImage       = "xai.view_image"
	toolViewXVideo      = "xai.view_x_video"
	toolImageGeneration = "xai.image_generation"
	toolFileSearch      = "xai.file_search"
	toolMCP             = "xai.mcp"
)

// Default names of the provider tools, which results use when the call did
// not declare the tool.
const (
	nameWebSearch       = "web_search"
	nameXSearch         = "x_search"
	nameCodeExecution   = "code_execution"
	nameViewImage       = "view_image"
	nameViewXVideo      = "view_x_video"
	nameImageGeneration = "image_generation"
	nameFileSearch      = "file_search"
	nameMCP             = "mcp"
)

// Search limits.
const (
	maxSearchDomains = 5
	maxXHandles      = 20
)

// WebSearchArgs configure the web search tool.
type WebSearchArgs struct {
	// AllowedDomains restricts the search to at most five domains.
	AllowedDomains []string `json:"allowedDomains,omitempty"`

	// ExcludedDomains excludes at most five domains.
	ExcludedDomains []string `json:"excludedDomains,omitempty"`

	EnableImageSearch        *bool `json:"enableImageSearch,omitempty"`
	EnableImageUnderstanding *bool `json:"enableImageUnderstanding,omitempty"`
}

// XSearchArgs configure the X search tool.
type XSearchArgs struct {
	// AllowedXHandles restricts the search to at most 20 handles.
	AllowedXHandles []string `json:"allowedXHandles,omitempty"`

	// ExcludedXHandles excludes at most 20 handles.
	ExcludedXHandles []string `json:"excludedXHandles,omitempty"`

	FromDate                 string `json:"fromDate,omitempty"`
	ToDate                   string `json:"toDate,omitempty"`
	EnableImageUnderstanding *bool  `json:"enableImageUnderstanding,omitempty"`
	EnableVideoUnderstanding *bool  `json:"enableVideoUnderstanding,omitempty"`
}

// ImageGenerationAction restricts the image generation tool.
type ImageGenerationAction string

// Image generation actions.
const (
	// ImageGenerationAuto lets the model generate and edit images. It is the
	// default.
	ImageGenerationAuto ImageGenerationAction = "auto"

	// ImageGenerationGenerate allows text-to-image generation only.
	ImageGenerationGenerate ImageGenerationAction = "generate"

	// ImageGenerationEdit allows image editing only.
	ImageGenerationEdit ImageGenerationAction = "edit"
)

// ImageGenerationArgs configure the image generation tool.
type ImageGenerationArgs struct {
	Action ImageGenerationAction `json:"action,omitempty"`
}

// FileSearchArgs configure the file search tool.
type FileSearchArgs struct {
	// VectorStoreIDs are the collection IDs to search.
	VectorStoreIDs []string `json:"vectorStoreIds"`

	// MaxNumResults limits the results.
	MaxNumResults *float64 `json:"maxNumResults,omitempty"`
}

// MCPServerArgs configure a remote MCP server tool.
type MCPServerArgs struct {
	ServerURL         string            `json:"serverUrl"`
	ServerLabel       string            `json:"serverLabel,omitempty"`
	ServerDescription string            `json:"serverDescription,omitempty"`
	AllowedTools      []string          `json:"allowedTools,omitempty"`
	Headers           map[string]string `json:"headers,omitempty"`
	Authorization     string            `json:"authorization,omitempty"`
}

// WebSearch returns the web search tool, which xAI executes. Its results are
// WebSearchResult values.
func WebSearch(args WebSearchArgs) model.ProviderTool {
	return newProviderTool(toolWebSearch, nameWebSearch, args)
}

// XSearch returns the X search tool, which xAI executes.
func XSearch(args XSearchArgs) model.ProviderTool {
	return newProviderTool(toolXSearch, nameXSearch, args)
}

// CodeExecution returns the code execution tool, which xAI executes.
func CodeExecution() model.ProviderTool {
	return newProviderTool(toolCodeExecution, nameCodeExecution, struct{}{})
}

// ViewImage returns the image viewing tool, which xAI executes.
func ViewImage() model.ProviderTool {
	return newProviderTool(toolViewImage, nameViewImage, struct{}{})
}

// ViewXVideo returns the X video viewing tool, which xAI executes.
func ViewXVideo() model.ProviderTool {
	return newProviderTool(toolViewXVideo, nameViewXVideo, struct{}{})
}

// ImageGeneration returns the image generation tool, which xAI executes. Its
// results are ImageGenerationResult values.
func ImageGeneration(args ImageGenerationArgs) model.ProviderTool {
	return newProviderTool(toolImageGeneration, nameImageGeneration, args)
}

// FileSearch returns the file search tool, which searches collections and
// which xAI executes. Its results are FileSearchResult values.
func FileSearch(args FileSearchArgs) model.ProviderTool {
	return newProviderTool(toolFileSearch, nameFileSearch, args)
}

// MCPServer returns a remote MCP server tool, which xAI executes.
func MCPServer(args MCPServerArgs) model.ProviderTool {
	return newProviderTool(toolMCP, nameMCP, args)
}

func newProviderTool[Args any](toolID string, name string, args Args) model.ProviderTool {
	encoded, err := json.Marshal(args)
	if err != nil {
		panic(fmt.Sprintf("encode %s arguments: %v", toolID, err))
	}

	return model.ProviderTool{
		ID:   toolID,
		Name: name,
		Args: encoded,
	}
}

// decodeToolArgs decodes the arguments of a provider tool. Absent arguments
// decode to the zero value.
func decodeToolArgs[Args any](tool model.ProviderTool) (Args, error) {
	var args Args
	if len(tool.Args) == 0 {
		return args, nil
	}

	if err := json.Unmarshal(tool.Args, &args); err != nil {
		return args, fmt.Errorf("%w: arguments of tool %s: %w", model.ErrInvalidArgument, tool.ID, err)
	}

	return args, nil
}

// WebSearchResult is the result of a web search call.
type WebSearchResult struct {
	Action  *WebSearchAction `json:"action,omitempty"`
	Sources []WebSource      `json:"sources,omitempty"`
}

// WebSearchAction is the action a web search call performed: "search",
// "openPage" or "findInPage".
type WebSearchAction struct {
	Type    string   `json:"type"`
	Query   *string  `json:"query,omitempty"`
	Queries []string `json:"queries,omitempty"`

	// URL is set, possibly to null, for openPage and findInPage actions.
	URL *nullableString `json:"url,omitempty"`

	// Pattern is set, possibly to null, for findInPage actions.
	Pattern *nullableString `json:"pattern,omitempty"`
}

// nullableString encodes a present value that may be null.
type nullableString struct {
	value *string
}

// MarshalJSON encodes the string or null.
func (s nullableString) MarshalJSON() ([]byte, error) {
	return json.Marshal(s.value)
}

// WebSource is a web page a search used.
type WebSource struct {
	Type string `json:"type"`
	URL  string `json:"url"`
}

// FileSearchResult is the result of a file search call.
type FileSearchResult struct {
	Queries []string `json:"queries"`

	// Results is null when the results were not included.
	Results *[]FileSearchMatch `json:"results"`
}

// FileSearchMatch is one retrieved file passage.
type FileSearchMatch struct {
	FileID   string  `json:"fileId"`
	Filename string  `json:"filename"`
	Score    float64 `json:"score"`
	Text     string  `json:"text"`
}

// ImageGenerationResult is the result of an image generation call.
type ImageGenerationResult struct {
	// Result is the generated image encoded in base64.
	Result string `json:"result"`

	// Prompt is the prompt the model wrote for the image model.
	Prompt *string `json:"prompt,omitempty"`
}
