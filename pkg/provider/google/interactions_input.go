package google

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	processingCallKind   = "google.processing_call"
	processingResultKind = "google.processing_result"

	executionDeniedResult = "Tool execution denied by user."
)

// convertedInteractionsInput is the converted prompt.
type convertedInteractionsInput struct {
	input             []interactionsStep
	systemInstruction string
	hasSystem         bool
	warnings          []model.Warning
}

// interactionsInputSettings control the prompt conversion.
type interactionsInputSettings struct {
	previousInteractionID string
	store                 *bool
	mediaResolution       InteractionsMediaResolution
}

// interactionsInputConverter converts one prompt into steps: user and tool
// turns become user_input steps, and assistant content fans out into
// model_output, thought, function_call and processing steps.
type interactionsInputConverter struct {
	settings interactionsInputSettings
	steps    []interactionsStep
	warnings []model.Warning
}

// convertInteractionsInput converts the prompt. With a previous
// interaction, assistant turns of that interaction are left out, unless the
// call is not stored, which makes the combination incoherent.
func convertInteractionsInput(prompt []model.Message, settings interactionsInputSettings) (convertedInteractionsInput, error) {
	converter := &interactionsInputConverter{
		settings: settings,
		steps:    []interactionsStep{},
	}

	notStored := settings.store != nil && !*settings.store
	if settings.previousInteractionID != "" && notStored {
		converter.warn("google.interactions: providerOptions.google.previousInteractionId was set together with store: false. " +
			"These are incoherent (the prior interaction cannot be referenced when nothing was stored on the server); " +
			"the full history will be sent and previous_interaction_id will still be emitted.")
	}

	if settings.previousInteractionID != "" && !notStored {
		compacted, err := compactInteractionsPrompt(prompt, settings.previousInteractionID)
		if err != nil {
			return convertedInteractionsInput{}, err
		}

		prompt = compacted
	}

	var systemTexts []string

	for _, message := range prompt {
		var err error

		switch typed := message.(type) {
		case model.SystemMessage:
			systemTexts = append(systemTexts, typed.Content)

		case model.UserMessage:
			err = converter.userMessage(typed)

		case model.AssistantMessage:
			err = converter.assistantMessage(typed)

		case model.ToolMessage:
			err = converter.toolMessage(typed)

		default:
			err = fmt.Errorf("%w: unsupported message %T", model.ErrInvalidPrompt, message)
		}

		if err != nil {
			return convertedInteractionsInput{}, err
		}
	}

	return convertedInteractionsInput{
		input:             converter.steps,
		systemInstruction: strings.Join(systemTexts, "\n\n"),
		hasSystem:         len(systemTexts) > 0,
		warnings:          converter.warnings,
	}, nil
}

func (c *interactionsInputConverter) warn(message string) {
	c.warnings = append(c.warnings, otherWarning(message))
}

func (c *interactionsInputConverter) userMessage(message model.UserMessage) error {
	var content []interactionsContent

	for _, part := range message.Content {
		switch typed := part.(type) {
		case model.TextPart:
			content = append(content, textContent(typed.Text))

		case model.FilePart:
			block, ok, err := c.fileContent(typed)
			if err != nil {
				return err
			}

			if ok {
				content = append(content, block)
			}
		}
	}

	merged := mergeAdjacentText(content)
	if len(merged) > 0 {
		c.steps = append(c.steps, interactionsStep{
			Type:    "user_input",
			Content: merged,
		})
	}

	return nil
}

// assistantMessage coalesces adjacent text and file content into one
// model_output step; every other part becomes its own step.
func (c *interactionsInputConverter) assistantMessage(message model.AssistantMessage) error {
	var pending []interactionsContent

	flush := func() {
		if len(pending) == 0 {
			return
		}

		c.steps = append(c.steps, interactionsStep{
			Type:    "model_output",
			Content: pending,
		})
		pending = nil
	}

	for _, part := range message.Content {
		switch typed := part.(type) {
		case model.TextPart:
			pending = append(pending, textContent(typed.Text))

		case model.FilePart:
			block, ok, err := c.fileContent(typed)
			if err != nil {
				return err
			}

			if ok {
				pending = append(pending, block)
			}

		case model.ReasoningPart:
			flush()

			step, err := thoughtStep(typed)
			if err != nil {
				return err
			}

			c.steps = append(c.steps, step)

		case model.CustomPart:
			flush()

			if err := c.processingStep(typed); err != nil {
				return err
			}

		case model.ToolCallPart:
			flush()

			step, err := functionCallStep(typed)
			if err != nil {
				return err
			}

			c.steps = append(c.steps, step)

		case model.ToolApprovalRequestPart:
			// Approval requests are conversation history only.

		default:
			c.warn(fmt.Sprintf("google.interactions: unsupported assistant content part type %q; part dropped.", assistantPartType(part)))
		}
	}

	flush()

	return nil
}

func thoughtStep(part model.ReasoningPart) (interactionsStep, error) {
	options, err := readPartOptions(part.ProviderOptions)
	if err != nil {
		return interactionsStep{}, err
	}

	step := interactionsStep{
		Type:      "thought",
		Signature: options.Signature,
	}

	if part.Text != "" {
		step.Summary = []interactionsSummaryItem{
			{
				Type: "text",
				Text: &part.Text,
			},
		}
	}

	return step, nil
}

// processingStep replays an agentic video processing call or result.
func (c *interactionsInputConverter) processingStep(part model.CustomPart) error {
	options, err := readPartOptions(part.ProviderOptions)
	if err != nil {
		return err
	}

	switch {
	case part.Kind == processingCallKind && options.ProcessingID != nil:
		c.steps = append(c.steps, interactionsStep{
			Type:      "processing_call",
			ID:        *options.ProcessingID,
			Signature: options.Signature,
		})

	case part.Kind == processingResultKind && options.ProcessingCallID != nil:
		c.steps = append(c.steps, interactionsStep{
			Type:      "processing_result",
			CallID:    *options.ProcessingCallID,
			Signature: options.Signature,
		})

	default:
		c.warn(fmt.Sprintf("google.interactions: unsupported or invalid custom assistant content part %q; part dropped.", part.Kind))
	}

	return nil
}

func functionCallStep(part model.ToolCallPart) (interactionsStep, error) {
	options, err := readPartOptions(part.ProviderOptions)
	if err != nil {
		return interactionsStep{}, err
	}

	arguments, err := toolCallArguments(part.Input)
	if err != nil {
		return interactionsStep{}, err
	}

	return interactionsStep{
		Type:      "function_call",
		ID:        part.ToolCallID,
		Name:      part.ToolName,
		Arguments: arguments,
		Signature: options.Signature,
	}, nil
}

// interactionsValueWrapper wraps arguments that are not an object.
type interactionsValueWrapper struct {
	Value json.RawMessage `json:"value"`
}

// toolCallArguments returns the arguments object. Other JSON values are
// wrapped under "value", and text that is not JSON as a string.
func toolCallArguments(input json.RawMessage) (json.RawMessage, error) {
	trimmed := bytes.TrimSpace(input)
	if len(trimmed) == 0 {
		return json.RawMessage("{}"), nil
	}

	if jsonobject.IsObject(trimmed) {
		return trimmed, nil
	}

	value := json.RawMessage(trimmed)
	if !json.Valid(value) {
		value = jsonobject.EncodeString(string(input))
	}

	encoded, err := json.Marshal(interactionsValueWrapper{
		Value: value,
	})
	if err != nil {
		return nil, fmt.Errorf("encode tool call arguments: %w", err)
	}

	return encoded, nil
}

// toolMessage sends the tool results as function_result blocks of one
// user_input step.
func (c *interactionsInputConverter) toolMessage(message model.ToolMessage) error {
	var content []interactionsContent

	for _, part := range message.Content {
		result, ok := part.(model.ToolResultPart)
		if !ok {
			c.warn(fmt.Sprintf("google.interactions: unsupported tool message part type %q; part dropped.", toolPartType(part)))

			continue
		}

		block, err := c.functionResult(result)
		if err != nil {
			return err
		}

		content = append(content, block)
	}

	if len(content) > 0 {
		c.steps = append(c.steps, interactionsStep{
			Type:    "user_input",
			Content: content,
		})
	}

	return nil
}

func (c *interactionsInputConverter) functionResult(part model.ToolResultPart) (interactionsContent, error) {
	options, err := readPartOptions(part.ProviderOptions)
	if err != nil {
		return interactionsContent{}, err
	}

	block := interactionsContent{
		Type:   "function_result",
		CallID: part.ToolCallID,
		Name:   part.ToolName,
	}

	if options.Signature != nil {
		block.Signature = *options.Signature
	}

	switch output := part.Output.(type) {
	case model.TextOutput:
		block.Result = jsonobject.EncodeString(output.Value)

	case model.JSONOutput:
		block.Result = jsonobject.EncodeString(jsonText(output.Value))

	case model.ErrorTextOutput:
		block.IsError = true
		block.Result = jsonobject.EncodeString(output.Value)

	case model.ErrorJSONOutput:
		block.IsError = true
		block.Result = jsonobject.EncodeString(jsonText(output.Value))

	case model.ExecutionDeniedOutput:
		reason := output.Reason
		if reason == "" {
			reason = executionDeniedResult
		}

		block.IsError = true
		block.Result = jsonobject.EncodeString(reason)

	case model.ContentOutput:
		result, err := c.contentResult(output)
		if err != nil {
			return interactionsContent{}, err
		}

		block.Result = result

	default:
		return interactionsContent{}, fmt.Errorf("%w: unsupported tool result output %T", model.ErrInvalidPrompt, part.Output)
	}

	return block, nil
}

// contentResult converts multimodal tool output into text and image
// blocks, which are all function results accept.
func (c *interactionsInputConverter) contentResult(output model.ContentOutput) (json.RawMessage, error) {
	blocks := []interactionsContent{}

	for _, item := range output.Value {
		switch typed := item.(type) {
		case model.ToolResultText:
			blocks = append(blocks, textContent(typed.Text))

		case model.ToolResultFile:
			if mediatype.TopLevel(typed.MediaType) != "image" {
				c.warn(fmt.Sprintf(
					"google.interactions: tool-result file with mediaType %q is not supported (Interactions `function_result.result` accepts only text and image content); part dropped.",
					typed.MediaType,
				))

				continue
			}

			block, ok, err := c.resultImage(typed)
			if err != nil {
				return nil, err
			}

			if ok {
				blocks = append(blocks, block)
			}

		default:
			c.warn(fmt.Sprintf("google.interactions: tool-result content part type %q is not supported; part dropped.", "custom"))
		}
	}

	encoded, err := json.Marshal(blocks)
	if err != nil {
		return nil, fmt.Errorf("encode tool result content: %w", err)
	}

	return encoded, nil
}

func (c *interactionsInputConverter) resultImage(file model.ToolResultFile) (interactionsContent, bool, error) {
	block := interactionsContent{
		Type: "image",
	}

	switch data := file.Data.(type) {
	case model.FileBytes:
		mediaType, err := mediatype.ResolveFile(file.MediaType, file.Data)
		if err != nil {
			return interactionsContent{}, false, err
		}

		block.Data = base64.StdEncoding.EncodeToString(data.Data)
		block.MimeType = mediaType

	case model.FileURL:
		block.URI = data.URL.String()
		block.MimeType = fullMediaTypeOrEmpty(file.MediaType)

	case model.FileReference:
		uri, err := googleReferenceURI(data)
		if err != nil {
			return interactionsContent{}, false, err
		}

		block.URI = uri
		block.MimeType = fullMediaTypeOrEmpty(file.MediaType)

	default:
		c.warn("google.interactions: tool-result image part with `data.type === \"text\"` is not representable as an image; part dropped.")

		return interactionsContent{}, false, nil
	}

	return block, true, nil
}

// fileContent converts a file part into an image, audio, video or document
// block, or text for text data. Other media types are dropped with a
// warning.
func (c *interactionsInputConverter) fileContent(file model.FilePart) (interactionsContent, bool, error) {
	if text, ok := file.Data.(model.FileText); ok {
		return textContent(text.Text), true, nil
	}

	kind, ok := interactionsFileKind(file.MediaType)
	if !ok {
		c.warn(fmt.Sprintf("google.interactions: unsupported file media type %q; part dropped.", file.MediaType))

		return interactionsContent{}, false, nil
	}

	block := interactionsContent{
		Type: kind,
	}

	if kind == "image" || kind == "video" {
		block.Resolution = c.settings.mediaResolution
	}

	if kind == "video" {
		processing, err := c.videoProcessing(file.ProviderOptions)
		if err != nil {
			return interactionsContent{}, false, err
		}

		block.Processing = processing
	}

	switch data := file.Data.(type) {
	case model.FileBytes:
		mediaType, err := mediatype.ResolveFile(file.MediaType, file.Data)
		if err != nil {
			return interactionsContent{}, false, err
		}

		block.Data = base64.StdEncoding.EncodeToString(data.Data)
		block.MimeType = mediaType

	case model.FileURL:
		block.URI = data.URL.String()
		block.MimeType = fullMediaTypeOrEmpty(file.MediaType)

	case model.FileReference:
		uri, err := googleReferenceURI(data)
		if err != nil {
			return interactionsContent{}, false, err
		}

		block.URI = uri
		block.MimeType = fullMediaTypeOrEmpty(file.MediaType)

	default:
		return interactionsContent{}, false, fmt.Errorf("%w: unsupported file data %T", model.ErrInvalidPrompt, file.Data)
	}

	return block, true, nil
}

// interactionsFileKind maps a top-level media type to a content block type.
func interactionsFileKind(mediaType string) (string, bool) {
	switch mediatype.TopLevel(mediaType) {
	case "image":
		return "image", true
	case "audio":
		return "audio", true
	case "video":
		return "video", true
	case "application", "text":
		return "document", true
	default:
		return "", false
	}
}

// videoProcessing converts the processing option of a video part. Invalid
// values are dropped with a warning.
func (c *interactionsInputConverter) videoProcessing(providerOptions model.ProviderOptions) (json.RawMessage, error) {
	options, err := readPartOptions(providerOptions)
	if err != nil {
		return nil, err
	}

	if !wirejson.HasValue(options.Processing) {
		return nil, nil
	}

	var mode string
	if err := json.Unmarshal(options.Processing, &mode); err == nil {
		if mode == string(InteractionsVideoAgentic) || mode == string(InteractionsVideoStatic) {
			return jsonobject.EncodeString(mode), nil
		}
	}

	var static interactionsStaticProcessing
	if err := json.Unmarshal(options.Processing, &static); err == nil && static.Type == string(InteractionsVideoStatic) {
		return encodeStaticProcessing(static)
	}

	c.warn("google.interactions: invalid providerOptions.google.processing on video file part; " +
		"expected \"agentic\", \"static\", or a static processing configuration. Option dropped.")

	return nil, nil
}

// encodeStaticProcessing writes the offsets as durations in seconds.
func encodeStaticProcessing(static interactionsStaticProcessing) (json.RawMessage, error) {
	wire := interactionsVideoStaticProcessing{
		Type: static.Type,
		FPS:  static.FPS,
	}

	if static.StartOffset != nil {
		wire.StartOffset = secondsText(*static.StartOffset)
	}

	if static.EndOffset != nil {
		wire.EndOffset = secondsText(*static.EndOffset)
	}

	encoded, err := json.Marshal(wire)
	if err != nil {
		return nil, fmt.Errorf("encode video processing: %w", err)
	}

	return encoded, nil
}

func secondsText(seconds float64) string {
	return strconv.FormatFloat(seconds, 'f', -1, 64) + "s"
}

// compactInteractionsPrompt drops the assistant turns of the previous
// interaction, and the tool results of their tool calls.
func compactInteractionsPrompt(prompt []model.Message, previousInteractionID string) ([]model.Message, error) {
	compacted := make([]model.Message, 0, len(prompt))
	droppedCalls := map[string]bool{}

	for _, message := range prompt {
		switch typed := message.(type) {
		case model.AssistantMessage:
			linked, err := isLinkedAssistantMessage(typed, previousInteractionID)
			if err != nil {
				return nil, err
			}

			if !linked {
				compacted = append(compacted, typed)

				continue
			}

			for _, part := range typed.Content {
				if call, ok := part.(model.ToolCallPart); ok {
					droppedCalls[call.ToolCallID] = true
				}
			}

		case model.ToolMessage:
			remaining := make([]model.ToolPart, 0, len(typed.Content))

			for _, part := range typed.Content {
				result, ok := part.(model.ToolResultPart)
				if ok && droppedCalls[result.ToolCallID] {
					continue
				}

				remaining = append(remaining, part)
			}

			if len(remaining) == 0 {
				continue
			}

			typed.Content = remaining
			compacted = append(compacted, typed)

		default:
			compacted = append(compacted, message)
		}
	}

	return compacted, nil
}

// isLinkedAssistantMessage reports whether any part of the message came
// from the interaction.
func isLinkedAssistantMessage(message model.AssistantMessage, interactionID string) (bool, error) {
	for _, part := range message.Content {
		options, err := readPartOptions(assistantPartOptions(part))
		if err != nil {
			return false, err
		}

		if options.InteractionID != nil && *options.InteractionID == interactionID {
			return true, nil
		}
	}

	return false, nil
}

func assistantPartOptions(part model.AssistantPart) model.ProviderOptions {
	switch typed := part.(type) {
	case model.TextPart:
		return typed.ProviderOptions
	case model.FilePart:
		return typed.ProviderOptions
	case model.ReasoningPart:
		return typed.ProviderOptions
	case model.ReasoningFilePart:
		return typed.ProviderOptions
	case model.CustomPart:
		return typed.ProviderOptions
	case model.ToolCallPart:
		return typed.ProviderOptions
	case model.ToolResultPart:
		return typed.ProviderOptions
	default:
		return nil
	}
}

func assistantPartType(part model.AssistantPart) string {
	switch part.(type) {
	case model.ReasoningFilePart:
		return "reasoning-file"
	case model.ToolResultPart:
		return "tool-result"
	default:
		return fmt.Sprintf("%T", part)
	}
}

func toolPartType(part model.ToolPart) string {
	if _, ok := part.(model.ToolApprovalResponsePart); ok {
		return "tool-approval-response"
	}

	return fmt.Sprintf("%T", part)
}

// mergeAdjacentText joins runs of text blocks of a user turn with a blank
// line. Blocks with annotations stay apart.
func mergeAdjacentText(content []interactionsContent) []interactionsContent {
	merged := make([]interactionsContent, 0, len(content))

	for _, block := range content {
		last := len(merged) - 1

		canMerge := last >= 0 &&
			isPlainText(merged[last]) &&
			isPlainText(block)

		if canMerge {
			merged[last] = textContent(*merged[last].Text + "\n\n" + *block.Text)

			continue
		}

		merged = append(merged, block)
	}

	return merged
}

func isPlainText(block interactionsContent) bool {
	return block.Type == "text" && block.Text != nil && block.Annotations == nil
}

func textContent(text string) interactionsContent {
	return interactionsContent{
		Type: "text",
		Text: &text,
	}
}

func readPartOptions(providerOptions model.ProviderOptions) (interactionsPartOptions, error) {
	options, _, err := model.ProviderValue[interactionsPartOptions](providerOptions, Namespace)

	return options, err
}

// googleReferenceURI returns the Google Files API URI of a reference.
func googleReferenceURI(reference model.FileReference) (string, error) {
	uri, ok := reference.IDs[Namespace]
	if !ok {
		return "", fmt.Errorf("%w: file reference has no %s file ID", model.ErrInvalidPrompt, Namespace)
	}

	return uri, nil
}

func fullMediaTypeOrEmpty(mediaType string) string {
	if mediatype.IsFull(mediaType) {
		return mediaType
	}

	return ""
}

// jsonText returns a JSON value as compact text, with "null" for a missing
// value.
func jsonText(value json.RawMessage) string {
	if len(bytes.TrimSpace(value)) == 0 {
		return "null"
	}

	var compacted bytes.Buffer
	if err := json.Compact(&compacted, value); err != nil {
		return string(value)
	}

	return compacted.String()
}
