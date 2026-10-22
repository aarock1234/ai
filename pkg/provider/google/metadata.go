package google

// ChatMetadata is the Google metadata of a language model response,
// stored under ProviderMetadata["google"].
type ChatMetadata struct {
	PromptFeedback    *PromptFeedback    `json:"promptFeedback"`
	GroundingMetadata *GroundingMetadata `json:"groundingMetadata"`

	// GroundingMetadataPerRequest holds the grounding of each automatically
	// continued request, in request order.
	GroundingMetadataPerRequest []*GroundingMetadata `json:"groundingMetadataPerRequest,omitempty"`

	URLContextMetadata *URLContextMetadata `json:"urlContextMetadata"`
	SafetyRatings      []SafetyRating      `json:"safetyRatings"`
	UsageMetadata      *UsageMetadata      `json:"usageMetadata"`
	FinishMessage      *string             `json:"finishMessage"`
	ServiceTier        *string             `json:"serviceTier"`

	// ContinuationToken is returned when a response stops with MAX_TOKENS
	// because it used up its MaxOutputTokens budget. Pass it back as
	// ChatOptions.ContinuationToken with a higher MaxOutputTokens
	// and the original prompt to resume generation.
	ContinuationToken string `json:"continuationToken,omitempty"`

	// InitialPromptTokenCount is the prompt token count of the first request
	// of an automatically continued response. Usage is summed across
	// requests, so use this value for context window checks. It is set only
	// when the response spanned more than one request.
	InitialPromptTokenCount *int `json:"initialPromptTokenCount,omitempty"`
}

// PartMetadata is the Google metadata of one content part. It is written to
// generated content and read back from replayed prompt parts.
type PartMetadata struct {
	// ThoughtSignature must be replayed with the part for multi-turn thinking
	// and function calling.
	ThoughtSignature string `json:"thoughtSignature,omitempty"`

	// ServerToolCallID and ServerToolType identify a server-side tool call
	// or response.
	ServerToolCallID string `json:"serverToolCallId,omitempty"`
	ServerToolType   string `json:"serverToolType,omitempty"`

	// Thought marks a replayed file part as part of the model's thinking.
	Thought bool `json:"thought,omitempty"`
}

// PromptFeedback reports whether the prompt was blocked.
type PromptFeedback struct {
	BlockReason   *string        `json:"blockReason,omitempty"`
	SafetyRatings []SafetyRating `json:"safetyRatings,omitempty"`
}

// SafetyRating is the safety assessment of one harm category.
type SafetyRating struct {
	Category         *string  `json:"category,omitempty"`
	Probability      *string  `json:"probability,omitempty"`
	ProbabilityScore *float64 `json:"probabilityScore,omitempty"`
	Severity         *string  `json:"severity,omitempty"`
	SeverityScore    *float64 `json:"severityScore,omitempty"`
	Blocked          *bool    `json:"blocked,omitempty"`
}

// URLContextMetadata reports the URLs retrieved by the URL context tool.
type URLContextMetadata struct {
	URLMetadata []URLMetadata `json:"urlMetadata,omitempty"`
}

// URLMetadata is the retrieval status of one URL.
type URLMetadata struct {
	RetrievedURL       string `json:"retrievedUrl"`
	URLRetrievalStatus string `json:"urlRetrievalStatus"`
}

// GroundingMetadata describes how a response was grounded.
type GroundingMetadata struct {
	WebSearchQueries   []string           `json:"webSearchQueries,omitempty"`
	ImageSearchQueries []string           `json:"imageSearchQueries,omitempty"`
	RetrievalQueries   []string           `json:"retrievalQueries,omitempty"`
	SearchEntryPoint   *SearchEntryPoint  `json:"searchEntryPoint,omitempty"`
	GroundingChunks    []GroundingChunk   `json:"groundingChunks,omitempty"`
	GroundingSupports  []GroundingSupport `json:"groundingSupports,omitempty"`
	RetrievalMetadata  *RetrievalMetadata `json:"retrievalMetadata,omitempty"`
}

// SearchEntryPoint is the rendered Google Search suggestion.
type SearchEntryPoint struct {
	RenderedContent string `json:"renderedContent"`
}

// GroundingChunk is one grounding source. Exactly one field is usually set.
type GroundingChunk struct {
	Web              *WebChunk              `json:"web,omitempty"`
	Image            *ImageChunk            `json:"image,omitempty"`
	RetrievedContext *RetrievedContextChunk `json:"retrievedContext,omitempty"`
	Maps             *MapsChunk             `json:"maps,omitempty"`
}

// WebChunk is a web page source.
type WebChunk struct {
	URI   string  `json:"uri"`
	Title *string `json:"title,omitempty"`
}

// ImageChunk is an image search source.
type ImageChunk struct {
	SourceURI string  `json:"sourceUri"`
	ImageURI  string  `json:"imageUri"`
	Title     *string `json:"title,omitempty"`
	Domain    *string `json:"domain,omitempty"`
}

// RetrievedContextChunk is a retrieval or file search source.
type RetrievedContextChunk struct {
	URI             *string          `json:"uri,omitempty"`
	Title           *string          `json:"title,omitempty"`
	Text            *string          `json:"text,omitempty"`
	FileSearchStore *string          `json:"fileSearchStore,omitempty"`
	CustomMetadata  []CustomMetadata `json:"customMetadata,omitempty"`
}

// CustomMetadata is a metadata entry of a retrieved document.
type CustomMetadata struct {
	Key             string           `json:"key"`
	StringValue     *string          `json:"stringValue,omitempty"`
	NumericValue    *float64         `json:"numericValue,omitempty"`
	StringListValue *StringListValue `json:"stringListValue,omitempty"`
}

// StringListValue is a list-valued metadata entry.
type StringListValue struct {
	Values []string `json:"values,omitempty"`
}

// MapsChunk is a Google Maps source.
type MapsChunk struct {
	URI     *string `json:"uri,omitempty"`
	Title   *string `json:"title,omitempty"`
	Text    *string `json:"text,omitempty"`
	PlaceID *string `json:"placeId,omitempty"`
}

// GroundingSupport links a segment of the response to grounding chunks.
type GroundingSupport struct {
	Segment               *Segment  `json:"segment,omitempty"`
	SegmentText           *string   `json:"segment_text,omitempty"`
	GroundingChunkIndices []int     `json:"groundingChunkIndices,omitempty"`
	SupportChunkIndices   []int     `json:"supportChunkIndices,omitempty"`
	ConfidenceScores      []float64 `json:"confidenceScores,omitempty"`
	ConfidenceScore       []float64 `json:"confidenceScore,omitempty"`
}

// Segment is a span of the response text.
type Segment struct {
	StartIndex *int    `json:"startIndex,omitempty"`
	EndIndex   *int    `json:"endIndex,omitempty"`
	Text       *string `json:"text,omitempty"`
}

// RetrievalMetadata reports dynamic retrieval scoring.
type RetrievalMetadata struct {
	WebDynamicRetrievalScore *float64 `json:"webDynamicRetrievalScore,omitempty"`
}

// UsageMetadata is the Gemini API token usage of a response.
type UsageMetadata struct {
	PromptTokenCount        *int    `json:"promptTokenCount,omitempty"`
	CandidatesTokenCount    *int    `json:"candidatesTokenCount,omitempty"`
	ToolUsePromptTokenCount *int    `json:"toolUsePromptTokenCount,omitempty"`
	TotalTokenCount         *int    `json:"totalTokenCount,omitempty"`
	CachedContentTokenCount *int    `json:"cachedContentTokenCount,omitempty"`
	ThoughtsTokenCount      *int    `json:"thoughtsTokenCount,omitempty"`
	TrafficType             *string `json:"trafficType,omitempty"`
	ServiceTier             *string `json:"serviceTier,omitempty"`

	PromptTokensDetails        []ModalityTokenCount `json:"promptTokensDetails,omitempty"`
	CacheTokensDetails         []ModalityTokenCount `json:"cacheTokensDetails,omitempty"`
	CandidatesTokensDetails    []ModalityTokenCount `json:"candidatesTokensDetails,omitempty"`
	ToolUsePromptTokensDetails []ModalityTokenCount `json:"toolUsePromptTokensDetails,omitempty"`
}

// ModalityTokenCount is the token count of one modality.
type ModalityTokenCount struct {
	Modality   string `json:"modality"`
	TokenCount int    `json:"tokenCount"`
}
