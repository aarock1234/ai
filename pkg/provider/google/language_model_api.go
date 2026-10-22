package google

import (
	"encoding/json"
)

// generateContentRequest is the body of generateContent and
// streamGenerateContent.
type generateContentRequest struct {
	GenerationConfig  generationConfig   `json:"generationConfig"`
	Contents          []content          `json:"contents"`
	SystemInstruction *systemInstruction `json:"systemInstruction,omitempty"`
	SafetySettings    []SafetySetting    `json:"safetySettings,omitempty"`
	Tools             []toolDeclaration  `json:"tools,omitempty"`
	ToolConfig        *toolConfig        `json:"toolConfig,omitempty"`
	CachedContent     string             `json:"cachedContent,omitempty"`
	Labels            map[string]string  `json:"labels,omitempty"`
	ServiceTier       ServiceTier        `json:"serviceTier,omitempty"`
	ContinuationToken string             `json:"continuationToken,omitempty"`
}

type generationConfig struct {
	MaxOutputTokens  *int     `json:"maxOutputTokens,omitempty"`
	Temperature      *float64 `json:"temperature,omitempty"`
	TopK             *int     `json:"topK,omitempty"`
	TopP             *float64 `json:"topP,omitempty"`
	FrequencyPenalty *float64 `json:"frequencyPenalty,omitempty"`
	PresencePenalty  *float64 `json:"presencePenalty,omitempty"`
	StopSequences    []string `json:"stopSequences,omitempty"`
	Seed             *int     `json:"seed,omitempty"`

	ResponseMimeType   string          `json:"responseMimeType,omitempty"`
	ResponseJSONSchema json.RawMessage `json:"responseJsonSchema,omitempty"`
	AudioTimestamp     bool            `json:"audioTimestamp,omitempty"`

	ResponseModalities []ResponseModality `json:"responseModalities,omitempty"`
	ThinkingConfig     *ThinkingConfig    `json:"thinkingConfig,omitempty"`
	MediaResolution    MediaResolution    `json:"mediaResolution,omitempty"`
	ImageConfig        *ImageConfig       `json:"imageConfig,omitempty"`
}

type systemInstruction struct {
	Parts []textPart `json:"parts"`
}

type textPart struct {
	Text string `json:"text"`
}

type contentRole string

const (
	roleUser  contentRole = "user"
	roleModel contentRole = "model"
)

type content struct {
	Role  contentRole `json:"role"`
	Parts []part      `json:"parts"`
}

// part is one content part in a request or response. Exactly one of the
// payload fields is set.
type part struct {
	Text                *string              `json:"text,omitempty"`
	InlineData          *InlineData          `json:"inlineData,omitempty"`
	FileData            *FileData            `json:"fileData,omitempty"`
	FunctionCall        *functionCall        `json:"functionCall,omitempty"`
	FunctionResponse    *functionResponse    `json:"functionResponse,omitempty"`
	ToolCall            *serverToolCall      `json:"toolCall,omitempty"`
	ToolResponse        *serverToolResponse  `json:"toolResponse,omitempty"`
	ExecutableCode      *CodeExecutionInput  `json:"executableCode,omitempty"`
	CodeExecutionResult *CodeExecutionOutput `json:"codeExecutionResult,omitempty"`

	Thought          bool   `json:"thought,omitempty"`
	ThoughtSignature string `json:"thoughtSignature,omitempty"`
}

// InlineData is inline file content encoded as base64.
type InlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

// FileData is file content referenced by URI.
type FileData struct {
	MimeType string `json:"mimeType"`
	FileURI  string `json:"fileUri"`
}

// functionCall is a client function call. Streaming responses may split its
// arguments into partial arguments.
type functionCall struct {
	ID           string            `json:"id,omitempty"`
	Name         string            `json:"name,omitempty"`
	Args         json.RawMessage   `json:"args,omitempty"`
	PartialArgs  []partialArgument `json:"partialArgs,omitempty"`
	WillContinue *bool             `json:"willContinue,omitempty"`
}

type functionResponse struct {
	ID       string                  `json:"id,omitempty"`
	Name     string                  `json:"name"`
	Response functionResponseContent `json:"response"`
	Parts    []functionResponsePart  `json:"parts,omitempty"`
}

// functionResponseContent carries a tool result as content or as an error.
// Both fields hold any JSON value.
type functionResponseContent struct {
	Name    string          `json:"name"`
	Content json.RawMessage `json:"content,omitempty"`
	Error   json.RawMessage `json:"error,omitempty"`
}

type functionResponsePart struct {
	InlineData *InlineData `json:"inlineData,omitempty"`
	FileData   *FileData   `json:"fileData,omitempty"`
}

// serverToolCall is a call of a server-side tool, such as Google Search,
// when it is combined with function tools.
type serverToolCall struct {
	ToolType string          `json:"toolType"`
	Args     json.RawMessage `json:"args,omitempty"`
	ID       string          `json:"id"`
}

type serverToolResponse struct {
	ToolType string          `json:"toolType"`
	Response json.RawMessage `json:"response,omitempty"`
	ID       string          `json:"id"`
}

type toolDeclaration struct {
	FunctionDeclarations []functionDeclaration `json:"functionDeclarations,omitempty"`
	GoogleSearch         *SearchArgs           `json:"googleSearch,omitempty"`
	EnterpriseWebSearch  *emptyObject          `json:"enterpriseWebSearch,omitempty"`
	URLContext           *emptyObject          `json:"urlContext,omitempty"`
	CodeExecution        *emptyObject          `json:"codeExecution,omitempty"`
	FileSearch           *FileSearchArgs       `json:"fileSearch,omitempty"`
	Retrieval            *retrieval            `json:"retrieval,omitempty"`
	GoogleMaps           *emptyObject          `json:"googleMaps,omitempty"`
}

// emptyObject encodes as {}.
type emptyObject struct{}

type functionDeclaration struct {
	Name                 string          `json:"name"`
	Description          string          `json:"description"`
	ParametersJSONSchema json.RawMessage `json:"parametersJsonSchema,omitempty"`
}

type retrieval struct {
	VertexRAGStore vertexRAGStore `json:"vertex_rag_store"`
}

type vertexRAGStore struct {
	RAGResources   ragResources `json:"rag_resources"`
	SimilarityTopK *int         `json:"similarity_top_k,omitempty"`
}

type ragResources struct {
	RAGCorpus string `json:"rag_corpus"`
}

type toolConfig struct {
	FunctionCallingConfig            *functionCallingConfig `json:"functionCallingConfig,omitempty"`
	IncludeServerSideToolInvocations bool                   `json:"includeServerSideToolInvocations,omitempty"`
	RetrievalConfig                  *RetrievalConfig       `json:"retrievalConfig,omitempty"`
}

type functionCallingMode string

const (
	functionCallingAuto      functionCallingMode = "AUTO"
	functionCallingNone      functionCallingMode = "NONE"
	functionCallingAny       functionCallingMode = "ANY"
	functionCallingValidated functionCallingMode = "VALIDATED"
)

type functionCallingConfig struct {
	Mode                 functionCallingMode `json:"mode,omitempty"`
	AllowedFunctionNames []string            `json:"allowedFunctionNames,omitempty"`

	// StreamFunctionCallArguments streams arguments on Vertex AI.
	StreamFunctionCallArguments bool `json:"streamFunctionCallArguments,omitempty"`
}

// generateContentResponse is a generateContent response or one stream chunk.
// Only stream chunks carry Error.
type generateContentResponse struct {
	Error          *Error          `json:"error,omitempty"`
	ResponseID     string          `json:"responseId,omitempty"`
	Candidates     []candidate     `json:"candidates,omitempty"`
	UsageMetadata  *UsageMetadata  `json:"usageMetadata,omitempty"`
	PromptFeedback *PromptFeedback `json:"promptFeedback,omitempty"`
}

type candidate struct {
	Content            *candidateContent   `json:"content,omitempty"`
	FinishReason       string              `json:"finishReason,omitempty"`
	FinishMessage      *string             `json:"finishMessage,omitempty"`
	ContinuationToken  string              `json:"continuationToken,omitempty"`
	SafetyRatings      []SafetyRating      `json:"safetyRatings,omitempty"`
	GroundingMetadata  *GroundingMetadata  `json:"groundingMetadata,omitempty"`
	URLContextMetadata *URLContextMetadata `json:"urlContextMetadata,omitempty"`
}

type candidateContent struct {
	Parts []part `json:"parts,omitempty"`
}

// firstCandidate returns the candidate that the SDK reports, if any.
func (r *generateContentResponse) firstCandidate() *candidate {
	if len(r.Candidates) == 0 {
		return nil
	}

	return &r.Candidates[0]
}

func (c *candidate) parts() []part {
	if c == nil || c.Content == nil {
		return nil
	}

	return c.Content.Parts
}
