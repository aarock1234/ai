package openresponses

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/internal/toolresult"
	"github.com/aarock1234/ai/pkg/model"
)

// Input item and part types.
const (
	typeMessage            = "message"
	typeReasoning          = "reasoning"
	typeFunctionCall       = "function_call"
	typeFunctionCallOutput = "function_call_output"
	typeCustomToolCall     = "custom_tool_call"
	typeCustomToolOutput   = "custom_tool_call_output"

	partInputText  = "input_text"
	partInputImage = "input_image"
	partInputFile  = "input_file"
	partOutputText = "output_text"

	defaultFilename = "data"
	statusCompleted = "completed"
)

// inputSettings configures prompt conversion.
type inputSettings struct {
	namespace           string
	strictResponseInput bool
	customToolID        string

	// providerTools maps tool names to the call's provider tools.
	providerTools map[string]model.ProviderTool
}

// convertedInput is the converted prompt.
type convertedInput struct {
	items        []inputItem
	instructions *string
	warnings     []model.Warning
}

// isCustomTool reports whether a tool name belongs to a custom tool.
func (s inputSettings) isCustomTool(toolName string) bool {
	tool, ok := s.providerTools[toolName]

	return ok && tool.ID == s.customToolID
}

// convertInput converts the prompt into input items. System messages become
// the instructions.
func convertInput(prompt []model.Message, settings inputSettings) (convertedInput, error) {
	converted := convertedInput{
		items: []inputItem{},
	}

	var systemMessages []string

	for _, message := range prompt {
		switch typed := message.(type) {
		case model.SystemMessage:
			systemMessages = append(systemMessages, typed.Content)

		case model.UserMessage:
			item, err := convertUserMessage(typed, settings)
			if err != nil {
				return convertedInput{}, err
			}

			converted.items = append(converted.items, item)

		case model.AssistantMessage:
			items, err := convertAssistantMessage(typed, settings, converted.items)
			if err != nil {
				return convertedInput{}, err
			}

			converted.items = items

		case model.ToolMessage:
			items, warnings, err := convertToolMessage(typed, settings)
			if err != nil {
				return convertedInput{}, err
			}

			converted.items = append(converted.items, items...)
			converted.warnings = append(converted.warnings, warnings...)

		case nil:
			return convertedInput{}, fmt.Errorf("%w: message is nil", model.ErrInvalidPrompt)

		default:
			return convertedInput{}, fmt.Errorf("%w: unsupported role %q", model.ErrInvalidPrompt, message.Role())
		}
	}

	if len(systemMessages) > 0 {
		converted.instructions = new(strings.Join(systemMessages, "\n"))
	}

	return converted, nil
}

func convertUserMessage(message model.UserMessage, settings inputSettings) (userMessage, error) {
	parts := []inputPart{}

	for _, part := range message.Content {
		switch typed := part.(type) {
		case model.TextPart:
			parts = append(parts, inputPart{
				Type: partInputText,
				Text: new(typed.Text),
			})

		case model.FilePart:
			converted, err := convertFilePart(typed, settings.namespace)
			if err != nil {
				return userMessage{}, err
			}

			parts = append(parts, converted)
		}
	}

	return userMessage{
		Type:    typeMessage,
		Role:    string(model.RoleUser),
		Content: parts,
	}, nil
}

// convertFilePart sends images as image parts and other files as file
// parts, by URL or inline.
func convertFilePart(part model.FilePart, namespace string) (inputPart, error) {
	switch part.Data.(type) {
	case model.FileReference:
		return inputPart{}, fmt.Errorf("%w: file parts with provider references", model.ErrUnsupported)

	case model.FileText:
		return inputPart{}, fmt.Errorf("%w: text file parts", model.ErrUnsupported)
	}

	return fileContent(fileContentSpec{
		mediaType: part.MediaType,
		filename:  part.Filename,
		data:      part.Data,
		detail:    readReplayData(part.ProviderOptions, namespace).imageDetail(),
	})
}

// fileContentSpec describes a file of a user message or a tool result.
type fileContentSpec struct {
	mediaType string
	filename  string
	data      model.FileData
	detail    ImageDetail
}

// fileContent converts a URL or inline file.
func fileContent(spec fileContentSpec) (inputPart, error) {
	isImage := mediatype.TopLevel(spec.mediaType) == "image"

	if location, ok := spec.data.(model.FileURL); ok {
		if location.URL == nil {
			return inputPart{}, fmt.Errorf("%w: file part URL is missing", model.ErrInvalidPrompt)
		}

		if isImage {
			return inputPart{
				Type:     partInputImage,
				ImageURL: location.URL.String(),
				Detail:   spec.detail,
			}, nil
		}

		return inputPart{
			Type:    partInputFile,
			FileURL: location.URL.String(),
		}, nil
	}

	inline, ok := spec.data.(model.FileBytes)
	if !ok {
		return inputPart{}, fmt.Errorf("%w: file part data type %T", model.ErrUnsupported, spec.data)
	}

	mediaType, err := mediatype.ResolveFile(spec.mediaType, inline)
	if err != nil {
		return inputPart{}, err
	}

	dataURL := dataurl.Encode(mediaType, inline.Data)

	if isImage {
		return inputPart{
			Type:     partInputImage,
			ImageURL: dataURL,
			Detail:   spec.detail,
		}, nil
	}

	filename := spec.filename
	if filename == "" {
		filename = defaultFilename
	}

	return inputPart{
		Type:     partInputFile,
		Filename: filename,
		FileData: dataURL,
	}, nil
}

// assistantWriter groups consecutive text of one message item into one
// assistant message.
type assistantWriter struct {
	settings inputSettings
	items    []inputItem

	content   []outputTextParam
	messageID *string
}

// flush appends the pending text as an assistant message.
func (w *assistantWriter) flush() {
	if len(w.content) == 0 {
		return
	}

	w.items = append(w.items, w.assistantMessage())
	w.content = nil
	w.messageID = nil
}

// assistantMessage sends an easy input message, or in strict mode a
// complete output item when the message has an item ID.
func (w *assistantWriter) assistantMessage() assistantMessage {
	message := assistantMessage{
		Type: typeMessage,
		Role: string(model.RoleAssistant),
	}

	if w.messageID != nil {
		message.ID = *w.messageID
	}

	if !w.settings.strictResponseInput {
		message.Content = assistantContent{
			parts: w.content,
		}

		return message
	}

	if w.messageID == nil {
		var text strings.Builder
		for _, part := range w.content {
			text.WriteString(part.Text)
		}

		message.Content = assistantContent{
			text: text.String(),
		}

		return message
	}

	parts := make([]outputTextParam, 0, len(w.content))
	for _, part := range w.content {
		if part.Annotations == nil {
			part.Annotations = &[]URLCitation{}
		}

		part.Logprobs = &[]json.RawMessage{}
		parts = append(parts, part)
	}

	message.Status = statusCompleted
	message.Content = assistantContent{
		parts: parts,
	}

	return message
}

// convertAssistantMessage appends the assistant message's items to items.
// Reasoning parts of the same item merge into the previous reasoning item.
func convertAssistantMessage(message model.AssistantMessage, settings inputSettings, items []inputItem) ([]inputItem, error) {
	writer := &assistantWriter{
		settings: settings,
		items:    items,
	}

	for _, part := range message.Content {
		switch typed := part.(type) {
		case model.ReasoningPart:
			writer.flush()
			writer.addReasoning(typed)

		case model.TextPart:
			writer.addText(typed)

		case model.ToolCallPart:
			writer.flush()

			item, err := toolCallItem(typed, settings)
			if err != nil {
				return nil, err
			}

			writer.items = append(writer.items, item)
		}
	}

	writer.flush()

	return writer.items, nil
}

func (w *assistantWriter) addText(part model.TextPart) {
	data := readReplayData(part.ProviderOptions, w.settings.namespace)
	itemID := data.itemID()

	if len(w.content) > 0 && !sameID(w.messageID, itemID) {
		w.flush()
	}

	w.messageID = itemID

	text := outputTextParam{
		Type: partOutputText,
		Text: part.Text,
	}

	if annotations := data.annotations(); annotations != nil {
		text.Annotations = &annotations
	}

	w.content = append(w.content, text)
}

func (w *assistantWriter) addReasoning(part model.ReasoningPart) {
	data := readReplayData(part.ProviderOptions, w.settings.namespace)

	item := &reasoningItem{
		ID:               data.itemID(),
		Type:             typeReasoning,
		Summary:          data.summary(),
		EncryptedContent: data.encryptedContent(),
	}

	if item.Summary == nil {
		item.Summary = []SummaryText{}
	}

	replayedContent := data.content()
	usesPartText := !data.hasReasoningContent() && part.Text != ""

	switch {
	case replayedContent != nil:
		item.Content = replayedContent

	case usesPartText:
		item.Content = []ReasoningText{
			{
				Type: reasoningTextType,
				Text: part.Text,
			},
		}
	}

	previous, hasPrevious := w.lastReasoning()
	continuesPrevious := hasPrevious && item.ID != nil && sameID(previous.ID, item.ID)

	if continuesPrevious {
		previous.Content = append(previous.Content, item.Content...)

		return
	}

	w.items = append(w.items, item)
}

// lastReasoning returns the last item when it is a reasoning item.
func (w *assistantWriter) lastReasoning() (*reasoningItem, bool) {
	if len(w.items) == 0 {
		return nil, false
	}

	item, ok := w.items[len(w.items)-1].(*reasoningItem)

	return item, ok
}

func sameID(left *string, right *string) bool {
	if left == nil || right == nil {
		return left == nil && right == nil
	}

	return *left == *right
}

// toolCallItem converts a tool call into a function call, or a custom tool
// call for custom tools.
func toolCallItem(part model.ToolCallPart, settings inputSettings) (inputItem, error) {
	arguments, err := toolInput(part.Input)
	if err != nil {
		return nil, err
	}

	itemID := readReplayData(part.ProviderOptions, settings.namespace).itemID()

	if settings.isCustomTool(part.ToolName) {
		return customToolCallItem{
			Type:   typeCustomToolCall,
			ID:     itemID,
			CallID: part.ToolCallID,
			Name:   part.ToolName,
			Input:  arguments,
		}, nil
	}

	return functionCallItem{
		Type:      typeFunctionCall,
		ID:        itemID,
		CallID:    part.ToolCallID,
		Name:      part.ToolName,
		Arguments: arguments,
	}, nil
}

// toolInput returns a string input unchanged and other input as JSON.
func toolInput(input json.RawMessage) (string, error) {
	if text := stringValue(input); text != nil {
		return *text, nil
	}

	return toolresult.CompactJSON(input)
}

// convertToolMessage converts tool results into tool outputs. Approval
// responses are skipped.
func convertToolMessage(message model.ToolMessage, settings inputSettings) ([]inputItem, []model.Warning, error) {
	var items []inputItem
	var warnings []model.Warning

	for _, part := range message.Content {
		result, ok := part.(model.ToolResultPart)
		if !ok {
			continue
		}

		output, outputWarnings, err := convertToolOutput(result.Output, settings.namespace)
		if err != nil {
			return nil, nil, err
		}

		warnings = append(warnings, outputWarnings...)

		outputType := typeFunctionCallOutput
		if settings.isCustomTool(result.ToolName) {
			outputType = typeCustomToolOutput
		}

		items = append(items, toolOutputItem{
			Type:   outputType,
			CallID: result.ToolCallID,
			Output: output,
		})
	}

	return items, warnings, nil
}

// convertToolOutput sends multipart results as content parts and other
// results as text.
func convertToolOutput(output model.ToolResultOutput, namespace string) (toolOutput, []model.Warning, error) {
	content, ok := output.(model.ContentOutput)
	if !ok {
		text, err := toolresult.Text(output)
		if err != nil {
			return toolOutput{}, nil, err
		}

		return toolOutput{
			text: text,
		}, nil, nil
	}

	parts := []inputPart{}
	var warnings []model.Warning

	for _, item := range content.Value {
		switch typed := item.(type) {
		case model.ToolResultText:
			parts = append(parts, inputPart{
				Type: partInputText,
				Text: new(typed.Text),
			})

		case model.ToolResultFile:
			part, warning, err := toolResultFile(typed, namespace)
			if err != nil {
				return toolOutput{}, nil, err
			}

			if warning != nil {
				warnings = append(warnings, *warning)

				continue
			}

			parts = append(parts, part)

		default:
			warnings = append(warnings, model.Warning{
				Type:    model.WarningOther,
				Message: "unsupported tool content part type: custom",
			})
		}
	}

	return toolOutput{
		parts: parts,
	}, warnings, nil
}

// toolResultFile converts a URL or inline file, or reports a warning for
// other file data.
func toolResultFile(item model.ToolResultFile, namespace string) (inputPart, *model.Warning, error) {
	switch data := item.Data.(type) {
	case model.FileURL, model.FileBytes:
		part, err := fileContent(fileContentSpec{
			mediaType: item.MediaType,
			filename:  item.Filename,
			data:      data,
			detail:    readReplayData(item.ProviderOptions, namespace).imageDetail(),
		})

		return part, nil, err

	default:
		return inputPart{}, &model.Warning{
			Type:    model.WarningOther,
			Message: fmt.Sprintf("unsupported tool content part type: file with data type: %s", fileDataType(data)),
		}, nil
	}
}

func fileDataType(data model.FileData) string {
	switch data.(type) {
	case model.FileReference:
		return "reference"

	case model.FileText:
		return "text"

	default:
		return fmt.Sprintf("%T", data)
	}
}
