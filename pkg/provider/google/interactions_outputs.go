package google

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/url"
	"slices"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/model"
)

// InteractionsMetadata is the Google metadata of an Interactions API
// response, stored under ProviderMetadata["google"].
type InteractionsMetadata struct {
	// InteractionID identifies the stored interaction. Pass it as
	// InteractionsOptions.PreviousInteractionID to chain the next turn.
	InteractionID string `json:"interactionId,omitempty"`

	// ServiceTier is the applied service tier.
	ServiceTier string `json:"serviceTier,omitempty"`

	// OutputTokensByModality counts output tokens per modality, such as
	// "video", when the API reports a breakdown.
	OutputTokensByModality map[string]int `json:"outputTokensByModality,omitempty"`
}

// InteractionsPartMetadata is the Google metadata of one Interactions API
// content part. It is written to generated content and read back from
// replayed prompt parts.
type InteractionsPartMetadata struct {
	// Signature must be replayed with thoughts and function calls.
	Signature string `json:"signature,omitempty"`

	// InteractionID is the interaction the part came from. Turns of the
	// previous interaction are left out of the next input.
	InteractionID string `json:"interactionId,omitempty"`

	// ProcessingID identifies an agentic video "google.processing_call".
	ProcessingID string `json:"processingId,omitempty"`

	// ProcessingCallID is the processing call of a
	// "google.processing_result".
	ProcessingCallID string `json:"processingCallId,omitempty"`
}

// metadata returns the part metadata, or nil when it is empty.
func (m InteractionsPartMetadata) metadata() model.ProviderMetadata {
	if m == (InteractionsPartMetadata{}) {
		return nil
	}

	return model.ProviderMetadata{
		Namespace: m,
	}
}

// knownDocumentMediaTypes map document extensions to media types.
var knownDocumentMediaTypes = []struct {
	extension string
	mediaType string
}{
	{
		extension: ".pdf",
		mediaType: "application/pdf",
	},
	{
		extension: ".txt",
		mediaType: "text/plain",
	},
	{
		extension: ".md",
		mediaType: "text/markdown",
	},
	{
		extension: ".markdown",
		mediaType: "text/markdown",
	},
	{
		extension: ".doc",
		mediaType: "application/msword",
	},
	{
		extension: ".docx",
		mediaType: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
	},
}

// interactionsOutputs converts the steps of a response.
type interactionsOutputs struct {
	generateID    func() string
	interactionID string

	content         []model.Content
	hasFunctionCall bool
}

// parseInteractionsOutputs converts the steps of a response into content.
// User input steps echo the input and are skipped.
func parseInteractionsOutputs(steps []interactionsStep, generateID func() string, interactionID string) ([]model.Content, bool, error) {
	outputs := &interactionsOutputs{
		generateID:    generateID,
		interactionID: interactionID,
	}

	for _, step := range steps {
		if err := outputs.step(step); err != nil {
			return nil, false, err
		}
	}

	return outputs.content, outputs.hasFunctionCall, nil
}

func (o *interactionsOutputs) step(step interactionsStep) error {
	switch step.Type {
	case "model_output":
		return o.modelOutput(step.Content)

	case "thought":
		o.content = append(o.content, model.Reasoning{
			Text: thoughtText(step.Summary),
			ProviderMetadata: InteractionsPartMetadata{
				Signature:     valueOrEmpty(step.Signature),
				InteractionID: o.interactionID,
			}.metadata(),
		})

	case "processing_call":
		o.content = append(o.content, model.Custom{
			Kind: processingCallKind,
			ProviderMetadata: InteractionsPartMetadata{
				Signature:     valueOrEmpty(step.Signature),
				InteractionID: o.interactionID,
				ProcessingID:  o.idOrGenerated(step.ID),
			}.metadata(),
		})

	case "processing_result":
		o.content = append(o.content, model.Custom{
			Kind: processingResultKind,
			ProviderMetadata: InteractionsPartMetadata{
				Signature:        valueOrEmpty(step.Signature),
				InteractionID:    o.interactionID,
				ProcessingCallID: o.idOrGenerated(step.CallID),
			}.metadata(),
		})

	case "function_call":
		o.hasFunctionCall = true

		o.content = append(o.content, model.ToolCall{
			ToolCallID: step.ID,
			ToolName:   step.Name,
			Input:      compactArguments(step.Arguments),
			ProviderMetadata: InteractionsPartMetadata{
				Signature:     valueOrEmpty(step.Signature),
				InteractionID: o.interactionID,
			}.metadata(),
		})

	default:
		o.builtinStep(step)
	}

	return nil
}

// builtinStep converts the provider-executed call or result of a built-in
// tool. Other steps are ignored.
func (o *interactionsOutputs) builtinStep(step interactionsStep) {
	switch {
	case slices.Contains(builtinToolCallSteps, step.Type):
		o.content = append(o.content, model.ToolCall{
			ToolCallID:       o.idOrGenerated(step.ID),
			ToolName:         builtinToolName(step.Type, step.Name),
			Input:            compactArguments(step.Arguments),
			ProviderExecuted: true,
		})

	case slices.Contains(builtinToolResultSteps, step.Type):
		o.content = append(o.content, model.ToolResult{
			ToolCallID: o.idOrGenerated(step.CallID),
			ToolName:   builtinToolName(step.Type, step.Name),
			Result:     resultOrNull(step.Result),
		})

		for _, source := range builtinResultSources(step.Type, step.Result, o.generateID) {
			o.content = append(o.content, source)
		}
	}
}

func (o *interactionsOutputs) modelOutput(blocks []interactionsContent) error {
	for _, block := range blocks {
		switch block.Type {
		case "text":
			o.content = append(o.content, model.Text{
				Text: valueOrEmpty(block.Text),
				ProviderMetadata: InteractionsPartMetadata{
					InteractionID: o.interactionID,
				}.metadata(),
			})

			for _, source := range annotationSources(block.Annotations, o.generateID) {
				o.content = append(o.content, source)
			}

		case "image", "video":
			file, ok, err := mediaFile(block.Type, block.Data, block.URI, block.MimeType, o.interactionID)
			if err != nil {
				return err
			}

			if ok {
				o.content = append(o.content, file)
			}
		}
	}

	return nil
}

func (o *interactionsOutputs) idOrGenerated(id string) string {
	if id != "" {
		return id
	}

	return o.generateID()
}

// thoughtText joins the text items of a thought summary.
func thoughtText(summary []interactionsSummaryItem) string {
	var texts []string

	for _, item := range summary {
		if item.Type == "text" && item.Text != nil {
			texts = append(texts, *item.Text)
		}
	}

	return strings.Join(texts, "\n")
}

// builtinToolName names a built-in tool after its step type, or after the
// reported name for MCP server tools.
func builtinToolName(stepType string, name string) string {
	if stepType == mcpServerToolCallStep || stepType == mcpServerToolResultStep {
		if name == "" {
			return mcpServerToolName
		}

		return name
	}

	return strings.TrimSuffix(strings.TrimSuffix(stepType, "_call"), "_result")
}

func resultOrNull(result json.RawMessage) json.RawMessage {
	if !wirejson.HasValue(result) {
		return json.RawMessage("null")
	}

	return result
}

// mediaFile converts generated image or video content, inline or by URI.
// It reports false for content with neither.
func mediaFile(kind string, data string, uri string, mimeType string, interactionID string) (model.File, bool, error) {
	mediaType := mimeType
	if mediaType == "" {
		mediaType = "image/png"
		if kind == "video" {
			mediaType = "video/mp4"
		}
	}

	file := model.File{
		MediaType: mediaType,
		ProviderMetadata: InteractionsPartMetadata{
			InteractionID: interactionID,
		}.metadata(),
	}

	switch {
	case data != "":
		decoded, err := base64.StdEncoding.DecodeString(data)
		if err != nil {
			return model.File{}, false, fmt.Errorf("%w: decode generated %s: %w", model.ErrInvalidResponse, kind, err)
		}

		file.Data = model.FileBytes{
			Data: decoded,
		}

	case uri != "":
		parsed, err := url.Parse(uri)
		if err != nil {
			return model.File{}, false, fmt.Errorf("%w: parse generated %s uri: %w", model.ErrInvalidResponse, kind, err)
		}

		file.Data = model.FileURL{
			URL: parsed,
		}

	default:
		return model.File{}, false, nil
	}

	return file, true, nil
}

// annotationSources converts the citations of a text block, once per URL
// or document.
func annotationSources(annotations []interactionsAnnotation, generateID func() string) []model.Source {
	seen := map[string]bool{}

	var sources []model.Source

	for _, annotation := range annotations {
		source, ok := annotationSource(annotation, generateID)
		if !ok || seen[sourceKey(source)] {
			continue
		}

		seen[sourceKey(source)] = true
		sources = append(sources, source)
	}

	return sources
}

func annotationSource(annotation interactionsAnnotation, generateID func() string) (model.Source, bool) {
	switch annotation.Type {
	case "url_citation":
		if annotation.URL == "" {
			return model.Source{}, false
		}

		return urlSource(annotation.URL, annotation.Title, generateID), true

	case "file_citation":
		return citedDocument(annotation.URL, annotation.DocumentURI, annotation.FileName, annotation.FileName, generateID)

	case "place_citation":
		if annotation.URL == "" {
			return model.Source{}, false
		}

		return urlSource(annotation.URL, annotation.Name, generateID), true

	default:
		return model.Source{}, false
	}
}

// citedDocument converts a file citation: a URL source for HTTP(S)
// locations and a document source otherwise.
func citedDocument(location string, documentURI string, fileName *string, title *string, generateID func() string) (model.Source, bool) {
	uri := firstNonEmpty(location, documentURI, valueOrEmpty(fileName))
	if uri == "" {
		return model.Source{}, false
	}

	if strings.HasPrefix(uri, "http://") || strings.HasPrefix(uri, "https://") {
		return urlSource(uri, title, generateID), true
	}

	filename := valueOrEmpty(fileName)
	if filename == "" {
		filename = lastPathSegment(uri)
	}

	documentTitle := firstNonEmpty(valueOrEmpty(title), valueOrEmpty(fileName), filename, uri)

	return documentSource(documentTitle, inferredDocumentMediaType(uri), filename, generateID), true
}

func inferredDocumentMediaType(uri string) string {
	lower := strings.ToLower(uri)

	for _, known := range knownDocumentMediaTypes {
		if strings.HasSuffix(lower, known.extension) {
			return known.mediaType
		}
	}

	return octetStreamMediaType
}

// sourceKey identifies a source for de-duplication.
func sourceKey(source model.Source) string {
	if source.SourceType == model.SourceURL {
		return "url:" + source.URL
	}

	return "doc:" + firstNonEmpty(source.Filename, source.Title)
}

// builtinResultEntry is one entry of a built-in tool result.
type builtinResultEntry struct {
	URL         string  `json:"url"`
	Title       *string `json:"title"`
	Status      *string `json:"status"`
	FileName    *string `json:"file_name"`
	DocumentURI string  `json:"document_uri"`

	Places []struct {
		Name *string `json:"name"`
		URL  string  `json:"url"`
	} `json:"places"`
}

// builtinResultSources converts the grounding sources of a built-in tool
// result: fetched URLs, search results, map places and file search
// documents. Results that are not lists of objects have no sources.
func builtinResultSources(stepType string, result json.RawMessage, generateID func() string) []model.Source {
	var rawEntries []json.RawMessage
	if err := json.Unmarshal(result, &rawEntries); err != nil {
		return nil
	}

	var sources []model.Source

	for _, rawEntry := range rawEntries {
		var entry builtinResultEntry
		if err := json.Unmarshal(rawEntry, &entry); err != nil {
			continue
		}

		sources = append(sources, entry.sources(stepType, generateID)...)
	}

	return sources
}

func (e builtinResultEntry) sources(stepType string, generateID func() string) []model.Source {
	switch stepType {
	case "url_context_result":
		if e.URL == "" || (e.Status != nil && *e.Status != "success") {
			return nil
		}

		return []model.Source{
			urlSource(e.URL, nil, generateID),
		}

	case "google_search_result":
		if e.URL == "" {
			return nil
		}

		return []model.Source{
			urlSource(e.URL, e.Title, generateID),
		}

	case "google_maps_result":
		var sources []model.Source

		for _, place := range e.Places {
			if place.URL != "" {
				sources = append(sources, urlSource(place.URL, place.Name, generateID))
			}
		}

		return sources

	case "file_search_result":
		source, ok := citedDocument(e.URL, e.DocumentURI, e.FileName, e.Title, generateID)
		if !ok {
			return nil
		}

		return []model.Source{
			source,
		}

	default:
		return nil
	}
}

// convertInteractionsUsage maps the usage. Thoughts count as reasoning
// output and cached tokens as cache reads.
func convertInteractionsUsage(raw json.RawMessage) (model.Usage, *interactionsUsage, error) {
	if !wirejson.HasValue(raw) {
		return model.Usage{}, nil, nil
	}

	var usage interactionsUsage
	if err := json.Unmarshal(raw, &usage); err != nil {
		return model.Usage{}, nil, fmt.Errorf("%w: decode interaction usage: %w", model.ErrInvalidResponse, err)
	}

	converted := model.Usage{
		InputTokens: model.InputTokens{
			Total:     usage.TotalInputTokens,
			CacheRead: usage.TotalCachedTokens,
		},
		OutputTokens: model.OutputTokens{
			Text:      usage.TotalOutputTokens,
			Reasoning: usage.TotalThoughtTokens,
		},
		Raw: raw,
	}

	if usage.TotalInputTokens != nil {
		converted.InputTokens.NoCache = new(*usage.TotalInputTokens - valueOrZero(usage.TotalCachedTokens))
	}

	if usage.TotalOutputTokens != nil || usage.TotalThoughtTokens != nil {
		converted.OutputTokens.Total = new(valueOrZero(usage.TotalOutputTokens) + valueOrZero(usage.TotalThoughtTokens))
	}

	return converted, &usage, nil
}

// outputTokensByModality returns the output token breakdown, or nil.
func outputTokensByModality(usage *interactionsUsage) map[string]int {
	if usage == nil || usage.OutputTokensByModality == nil {
		return nil
	}

	breakdown := map[string]int{}

	for _, entry := range usage.OutputTokensByModality {
		if entry.Modality != nil && entry.Tokens != nil {
			breakdown[*entry.Modality] = *entry.Tokens
		}
	}

	if len(breakdown) == 0 {
		return nil
	}

	return breakdown
}

// interactionsFinishReason maps the interaction status. Client function
// calls finish with tool calls.
func interactionsFinishReason(status string, hasFunctionCall bool) model.FinishReason {
	unified := model.FinishOther

	switch status {
	case interactionCompleted:
		unified = model.FinishStop
		if hasFunctionCall {
			unified = model.FinishToolCalls
		}

	case interactionRequiresAction:
		unified = model.FinishToolCalls

	case interactionFailed:
		unified = model.FinishError

	case interactionIncomplete:
		unified = model.FinishLength
	}

	return model.FinishReason{
		Unified: unified,
		Raw:     status,
	}
}

// interactionsTimestamp parses the creation time, or returns zero.
func interactionsTimestamp(created *string) time.Time {
	if created == nil {
		return time.Time{}
	}

	parsed, err := time.Parse(time.RFC3339Nano, *created)
	if err != nil {
		return time.Time{}
	}

	return parsed
}

// nonEmptyID returns the interaction ID, which unstored interactions omit
// or send empty.
func nonEmptyID(id *string) string {
	if id == nil {
		return ""
	}

	return *id
}
