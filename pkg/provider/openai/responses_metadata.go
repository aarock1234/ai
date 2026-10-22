package openai

// ResponsesMetadata is the response-level provider metadata of a Responses
// call.
type ResponsesMetadata struct {
	// ResponseID identifies the response, for example to continue with
	// PreviousResponseID.
	ResponseID string `json:"responseId"`

	// Logprobs holds one list of token log probabilities per text part when
	// they were requested.
	Logprobs [][]ResponsesLogprob `json:"logprobs,omitempty"`

	// ServiceTier is the tier that processed the request.
	ServiceTier string `json:"serviceTier,omitempty"`

	// ReasoningContext is the reasoning context the model used.
	ReasoningContext string `json:"reasoningContext,omitempty"`
}

// ResponsesLogprob is the log probability of one generated token.
type ResponsesLogprob struct {
	Token       string                `json:"token"`
	Logprob     float64               `json:"logprob"`
	TopLogprobs []ResponsesTopLogprob `json:"top_logprobs"`
}

// ResponsesTopLogprob is one likely alternative for a generated token.
type ResponsesTopLogprob struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
}

// ResponsesReasoningMetadata is the provider metadata of reasoning content.
// It is read back from reasoning parts when the conversation continues.
type ResponsesReasoningMetadata struct {
	// ItemID identifies the reasoning item.
	ItemID string `json:"itemId,omitempty"`

	// ReasoningEncryptedContent is the encrypted reasoning, returned when
	// responses are not stored.
	ReasoningEncryptedContent *string `json:"reasoningEncryptedContent,omitempty"`
}

// ResponsesTextMetadata is the provider metadata of text content.
type ResponsesTextMetadata struct {
	// ItemID identifies the message item.
	ItemID string `json:"itemId,omitempty"`

	// Phase is "commentary" or "final_answer" when the model reports it.
	Phase string `json:"phase,omitempty"`

	// Annotations are the citations and file paths attached to the text.
	Annotations []ResponsesAnnotation `json:"annotations,omitempty"`
}

// ResponsesAnnotationType identifies the kind of text annotation.
type ResponsesAnnotationType string

// Annotation types.
const (
	ResponsesURLCitation           ResponsesAnnotationType = "url_citation"
	ResponsesFileCitation          ResponsesAnnotationType = "file_citation"
	ResponsesContainerFileCitation ResponsesAnnotationType = "container_file_citation"
	ResponsesFilePath              ResponsesAnnotationType = "file_path"
)

// ResponsesAnnotation is a text annotation in the API's format. URL
// citations set the indexes, URL and title; file citations set the file ID,
// filename and index; container file citations set the container ID, file
// ID, filename and indexes; file paths set the file ID and index.
type ResponsesAnnotation struct {
	Type        ResponsesAnnotationType `json:"type"`
	StartIndex  *int                    `json:"start_index,omitempty"`
	EndIndex    *int                    `json:"end_index,omitempty"`
	URL         string                  `json:"url,omitempty"`
	Title       string                  `json:"title,omitempty"`
	ContainerID string                  `json:"container_id,omitempty"`
	FileID      string                  `json:"file_id,omitempty"`
	Filename    string                  `json:"filename,omitempty"`
	Index       *int                    `json:"index,omitempty"`
}

// ResponsesToolCallMetadata is the provider metadata of a tool call. It is
// read back from tool call parts when the conversation continues.
type ResponsesToolCallMetadata struct {
	// ItemID identifies the output item that holds the call.
	ItemID string `json:"itemId,omitempty"`

	// Async reports that the model continued without waiting for the result.
	Async *bool `json:"async,omitempty"`

	// Namespace is the tool namespace of a namespaced function.
	Namespace string `json:"namespace,omitempty"`

	// Caller identifies the program that made the call.
	Caller *ResponsesToolCaller `json:"caller,omitempty"`

	// ParallelToolCall links a call expanded from a parallel wrapper call to
	// its original.
	ParallelToolCall *ResponsesParallelToolCall `json:"parallelToolCall,omitempty"`

	// ApprovalRequestID links a dynamic MCP tool call to its approval
	// request.
	ApprovalRequestID string `json:"approvalRequestId,omitempty"`
}

// ResponsesToolCallerType identifies who called a tool.
type ResponsesToolCallerType string

// Tool caller types.
const (
	ResponsesCallerDirect  ResponsesToolCallerType = "direct"
	ResponsesCallerProgram ResponsesToolCallerType = "program"
)

// ResponsesToolCaller identifies who called a tool.
type ResponsesToolCaller struct {
	Type ResponsesToolCallerType `json:"type"`

	// CallerID identifies the program when Type is "program".
	CallerID string `json:"callerId,omitempty"`
}

// ResponsesParallelToolCall records the parallel wrapper call that a tool
// call was expanded from, so that the results can be sent back as the
// wrapper's single output.
type ResponsesParallelToolCall struct {
	ItemID     string `json:"itemId"`
	ToolCallID string `json:"toolCallId"`
	ToolName   string `json:"toolName"`
	Input      string `json:"input"`
	Index      int    `json:"index"`
	Count      int    `json:"count"`
}

// ResponsesItemMetadata identifies the output item of provider-executed tool
// content.
type ResponsesItemMetadata struct {
	ItemID string `json:"itemId"`
}

// ResponsesCompactionMetadata is the provider metadata of custom
// "openai.compaction" content.
type ResponsesCompactionMetadata struct {
	// Type is "compaction".
	Type             string `json:"type"`
	ItemID           string `json:"itemId"`
	EncryptedContent string `json:"encryptedContent,omitempty"`
}

// ResponsesSourceMetadata is the provider metadata of document sources. Type
// is "file_search", "file_citation", "container_file_citation" or
// "file_path".
type ResponsesSourceMetadata struct {
	Type        string `json:"type"`
	FileID      string `json:"fileId"`
	Index       *int   `json:"index,omitempty"`
	ContainerID string `json:"containerId,omitempty"`
}
