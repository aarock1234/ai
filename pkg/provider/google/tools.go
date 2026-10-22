package google

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/model"
)

// Provider tool IDs.
const (
	googleSearchToolID        = "google.google_search"
	enterpriseWebSearchToolID = "google.enterprise_web_search"
	googleMapsToolID          = "google.google_maps"
	urlContextToolID          = "google.url_context"
	fileSearchToolID          = "google.file_search"
	codeExecutionToolID       = "google.code_execution"
	vertexRAGStoreToolID      = "google.vertex_rag_store"
)

// codeExecutionToolName is the wire name of the code execution tool.
const codeExecutionToolName = "code_execution"

// SearchArgs configures the Google Search tool.
type SearchArgs struct {
	SearchTypes     *SearchTypes     `json:"searchTypes,omitempty"`
	TimeRangeFilter *TimeRangeFilter `json:"timeRangeFilter,omitempty"`
}

// SearchTypes enables web and image search. Set a field to a non-nil value
// to enable that search type.
type SearchTypes struct {
	WebSearch   *struct{} `json:"webSearch,omitempty"`
	ImageSearch *struct{} `json:"imageSearch,omitempty"`
}

// TimeRangeFilter restricts search results to a time range.
type TimeRangeFilter struct {
	StartTime string `json:"startTime"`
	EndTime   string `json:"endTime"`
}

// FileSearchArgs configures the File Search tool.
type FileSearchArgs struct {
	// FileSearchStoreNames are fully qualified store names, such as
	// "fileSearchStores/my-file-search-store-123".
	FileSearchStoreNames []string `json:"fileSearchStoreNames"`

	// TopK optionally limits the number of retrieved chunks.
	TopK *int `json:"topK,omitempty"`

	// MetadataFilter optionally restricts the retrieved documents, using the
	// syntax of https://google.aip.dev/160.
	MetadataFilter string `json:"metadataFilter,omitempty"`
}

// VertexRAGStoreArgs configures the Vertex RAG Store tool.
type VertexRAGStoreArgs struct {
	// RAGCorpus is a corpus resource name, such as
	// "projects/{project}/locations/{location}/ragCorpora/{rag_corpus}".
	RAGCorpus string `json:"ragCorpus"`

	// TopK optionally limits the number of retrieved contexts.
	TopK *int `json:"topK,omitempty"`
}

// CodeExecutionInput is the input of a code execution tool call.
type CodeExecutionInput struct {
	Language string `json:"language"`
	Code     string `json:"code"`
}

// CodeExecutionOutput is the result of a code execution tool call.
type CodeExecutionOutput struct {
	// Outcome is the execution outcome, such as "OUTCOME_OK".
	Outcome string `json:"outcome"`
	Output  string `json:"output"`
}

// Search returns the Google Search grounding tool, named
// "google_search". It requires Gemini 2.0 or newer.
func Search(args SearchArgs) model.ProviderTool {
	return providerTool(googleSearchToolID, "google_search", args)
}

// EnterpriseWebSearch returns the Enterprise Web Search grounding tool, named
// "enterprise_web_search". It uses a compliance-focused web index and is
// available on Vertex AI only.
func EnterpriseWebSearch() model.ProviderTool {
	return providerTool(enterpriseWebSearchToolID, "enterprise_web_search", emptyObject{})
}

// Maps returns the Google Maps grounding tool, named "google_maps".
func Maps() model.ProviderTool {
	return providerTool(googleMapsToolID, "google_maps", emptyObject{})
}

// URLContext returns the URL context tool, named "url_context". The model
// reads the URLs contained in the prompt.
func URLContext() model.ProviderTool {
	return providerTool(urlContextToolID, "url_context", emptyObject{})
}

// FileSearch returns the File Search retrieval tool, named "file_search". It
// requires Gemini 2.5 or newer.
func FileSearch(args FileSearchArgs) model.ProviderTool {
	return providerTool(fileSearchToolID, "file_search", args)
}

// CodeExecution returns the code execution tool, named "code_execution". The
// model generates and runs Python code; calls and results are reported as
// provider-executed tool calls with CodeExecutionInput and
// CodeExecutionOutput values.
func CodeExecution() model.ProviderTool {
	return providerTool(codeExecutionToolID, codeExecutionToolName, emptyObject{})
}

// VertexRAGStore returns the Vertex RAG Store retrieval tool, named
// "vertex_rag_store". It works with Vertex AI only.
func VertexRAGStore(args VertexRAGStoreArgs) model.ProviderTool {
	return providerTool(vertexRAGStoreToolID, "vertex_rag_store", args)
}

func providerTool[Args any](id string, name string, args Args) model.ProviderTool {
	encoded, err := json.Marshal(args)
	if err != nil {
		// The argument types are plain structs, which always encode.
		panic(fmt.Sprintf("encode %s tool arguments: %v", id, err))
	}

	return model.ProviderTool{
		ID:   id,
		Name: name,
		Args: encoded,
	}
}

// decodeToolArgs decodes the arguments of a provider tool. Missing arguments
// decode as the zero value.
func decodeToolArgs[T any](tool model.ProviderTool) (T, error) {
	var args T
	if !wirejson.HasValue(tool.Args) {
		return args, nil
	}

	if err := json.Unmarshal(tool.Args, &args); err != nil {
		return args, fmt.Errorf("%w: decode %s tool arguments: %w", model.ErrInvalidArgument, tool.ID, err)
	}

	return args, nil
}

func (a FileSearchArgs) validate() error {
	if a.FileSearchStoreNames == nil {
		return fmt.Errorf("%w: file search requires fileSearchStoreNames", model.ErrInvalidArgument)
	}

	if a.TopK != nil && *a.TopK <= 0 {
		return fmt.Errorf("%w: file search topK must be positive", model.ErrInvalidArgument)
	}

	return nil
}
