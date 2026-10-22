package gateway

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

// Part types of prompt messages and tool result content.
const (
	partText                 = "text"
	partFile                 = "file"
	partReasoning            = "reasoning"
	partReasoningFile        = "reasoning-file"
	partCustom               = "custom"
	partToolCall             = "tool-call"
	partToolResult           = "tool-result"
	partToolApprovalResponse = "tool-approval-response"
)

// Tool result output types.
const (
	outputText            = "text"
	outputJSON            = "json"
	outputExecutionDenied = "execution-denied"
	outputErrorText       = "error-text"
	outputErrorJSON       = "error-json"
	outputContent         = "content"
)

// callOptions are the language model call options on the wire, in the
// specification's JSON form.
type callOptions struct {
	Prompt           []promptMessage       `json:"prompt"`
	MaxOutputTokens  *int                  `json:"maxOutputTokens,omitempty"`
	Temperature      *float64              `json:"temperature,omitempty"`
	StopSequences    []string              `json:"stopSequences,omitempty"`
	TopP             *float64              `json:"topP,omitempty"`
	TopK             *int                  `json:"topK,omitempty"`
	PresencePenalty  *float64              `json:"presencePenalty,omitempty"`
	FrequencyPenalty *float64              `json:"frequencyPenalty,omitempty"`
	ResponseFormat   *responseFormat       `json:"responseFormat,omitempty"`
	Seed             *int                  `json:"seed,omitempty"`
	Tools            []tool                `json:"tools,omitempty"`
	ToolChoice       *toolChoice           `json:"toolChoice,omitempty"`
	IncludeRawChunks bool                  `json:"includeRawChunks,omitempty"`
	Headers          map[string]string     `json:"headers,omitempty"`
	Reasoning        model.ReasoningEffort `json:"reasoning,omitempty"`
	ProviderOptions  model.ProviderOptions `json:"providerOptions,omitempty"`
}

type responseFormat struct {
	Type        model.ResponseFormatType `json:"type"`
	Schema      json.RawMessage          `json:"schema,omitempty"`
	Name        string                   `json:"name,omitempty"`
	Description string                   `json:"description,omitempty"`
}

type toolChoice struct {
	Type     model.ToolChoiceType `json:"type"`
	ToolName string               `json:"toolName,omitempty"`
}

// tool is a function or provider tool on the wire.
type tool interface {
	isTool()
}

type functionTool struct {
	Type            string                `json:"type"`
	Name            string                `json:"name"`
	Description     string                `json:"description,omitempty"`
	InputSchema     json.RawMessage       `json:"inputSchema"`
	InputExamples   []inputExample        `json:"inputExamples,omitempty"`
	Strict          *bool                 `json:"strict,omitempty"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

type inputExample struct {
	Input json.RawMessage `json:"input"`
}

type providerTool struct {
	Type string          `json:"type"`
	ID   string          `json:"id"`
	Name string          `json:"name"`
	Args json.RawMessage `json:"args"`
}

func (functionTool) isTool() {}
func (providerTool) isTool() {}

// promptMessage is a system message or a message with parts.
type promptMessage interface {
	isPromptMessage()
}

type systemMessage struct {
	Role            model.Role            `json:"role"`
	Content         string                `json:"content"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

type partsMessage struct {
	Role            model.Role            `json:"role"`
	Content         []promptPart          `json:"content"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

func (systemMessage) isPromptMessage() {}
func (partsMessage) isPromptMessage()  {}

// promptPart is one part of a message on the wire.
type promptPart interface {
	isPromptPart()
}

// textPart is a text or reasoning part.
type textPart struct {
	Type            string                `json:"type"`
	Text            string                `json:"text"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

// filePart is a file or reasoning file part.
type filePart struct {
	Type            string                `json:"type"`
	Filename        string                `json:"filename,omitempty"`
	Data            wireFileData          `json:"data"`
	MediaType       string                `json:"mediaType"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

type customPart struct {
	Type            string                `json:"type"`
	Kind            string                `json:"kind"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

type toolCallPart struct {
	Type             string                `json:"type"`
	ToolCallID       string                `json:"toolCallId"`
	ToolName         string                `json:"toolName"`
	Input            json.RawMessage       `json:"input"`
	ProviderExecuted bool                  `json:"providerExecuted,omitempty"`
	ProviderOptions  model.ProviderOptions `json:"providerOptions,omitempty"`
}

type toolResultPart struct {
	Type            string                `json:"type"`
	ToolCallID      string                `json:"toolCallId"`
	ToolName        string                `json:"toolName"`
	Output          toolResultOutput      `json:"output"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

type toolApprovalResponsePart struct {
	Type            string                `json:"type"`
	ApprovalID      string                `json:"approvalId"`
	Approved        bool                  `json:"approved"`
	Reason          string                `json:"reason,omitempty"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

func (textPart) isPromptPart()                 {}
func (filePart) isPromptPart()                 {}
func (customPart) isPromptPart()               {}
func (toolCallPart) isPromptPart()             {}
func (toolResultPart) isPromptPart()           {}
func (toolApprovalResponsePart) isPromptPart() {}

// toolResultOutput is a tool result output on the wire. Value is absent
// for denied executions.
type toolResultOutput struct {
	Type            string                `json:"type"`
	Value           json.RawMessage       `json:"value,omitempty"`
	Reason          string                `json:"reason,omitempty"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

// outputItem is one item of a content tool result.
type outputItem interface {
	isOutputItem()
}

type outputTextItem struct {
	Type            string                `json:"type"`
	Text            string                `json:"text"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

type outputFileItem struct {
	Type            string                `json:"type"`
	Data            wireFileData          `json:"data"`
	MediaType       string                `json:"mediaType"`
	Filename        string                `json:"filename,omitempty"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

type outputCustomItem struct {
	Type            string                `json:"type"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

func (outputTextItem) isOutputItem()   {}
func (outputFileItem) isOutputItem()   {}
func (outputCustomItem) isOutputItem() {}

// encodeCallOptions converts the call options to their wire form. Inline
// file bytes become base64.
func encodeCallOptions(options model.CallOptions) (callOptions, error) {
	prompt, err := encodePrompt(options.Prompt)
	if err != nil {
		return callOptions{}, err
	}

	tools, err := encodeTools(options.Tools)
	if err != nil {
		return callOptions{}, err
	}

	encoded := callOptions{
		Prompt:           prompt,
		MaxOutputTokens:  options.MaxOutputTokens,
		Temperature:      options.Temperature,
		StopSequences:    options.StopSequences,
		TopP:             options.TopP,
		TopK:             options.TopK,
		PresencePenalty:  options.PresencePenalty,
		FrequencyPenalty: options.FrequencyPenalty,
		Seed:             options.Seed,
		Tools:            tools,
		IncludeRawChunks: options.IncludeRawChunks,
		Headers:          flattenHeaders(options.Headers),
		Reasoning:        options.Reasoning,
		ProviderOptions:  options.ProviderOptions,
	}

	if options.ResponseFormat != nil {
		encoded.ResponseFormat = &responseFormat{
			Type:        options.ResponseFormat.Type,
			Schema:      options.ResponseFormat.Schema,
			Name:        options.ResponseFormat.Name,
			Description: options.ResponseFormat.Description,
		}
	}

	if options.ToolChoice != nil {
		encoded.ToolChoice = &toolChoice{
			Type:     options.ToolChoice.Type,
			ToolName: options.ToolChoice.ToolName,
		}
	}

	return encoded, nil
}

// flattenHeaders joins repeated header values, as a header record holds one
// value per name.
func flattenHeaders(headers http.Header) map[string]string {
	if len(headers) == 0 {
		return nil
	}

	flattened := make(map[string]string, len(headers))
	for name, values := range headers {
		flattened[name] = strings.Join(values, ", ")
	}

	return flattened
}

func encodeTools(tools []model.Tool) ([]tool, error) {
	if len(tools) == 0 {
		return nil, nil
	}

	encoded := make([]tool, 0, len(tools))

	for _, candidate := range tools {
		switch typed := candidate.(type) {
		case model.FunctionTool:
			encoded = append(encoded, encodeFunctionTool(typed))

		case model.ProviderTool:
			encoded = append(encoded, providerTool{
				Type: "provider",
				ID:   typed.ID,
				Name: typed.Name,
				Args: objectOrEmpty(typed.Args),
			})

		default:
			return nil, fmt.Errorf("%w: tool %T", model.ErrUnsupported, candidate)
		}
	}

	return encoded, nil
}

func encodeFunctionTool(function model.FunctionTool) functionTool {
	examples := make([]inputExample, 0, len(function.InputExamples))
	for _, example := range function.InputExamples {
		examples = append(examples, inputExample{
			Input: example,
		})
	}

	return functionTool{
		Type:            "function",
		Name:            function.Name,
		Description:     function.Description,
		InputSchema:     objectOrEmpty(function.InputSchema),
		InputExamples:   examples,
		Strict:          function.Strict,
		ProviderOptions: function.ProviderOptions,
	}
}

// objectOrEmpty returns raw, or an empty object when it is absent.
func objectOrEmpty(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return json.RawMessage("{}")
	}

	return raw
}

func encodePrompt(prompt []model.Message) ([]promptMessage, error) {
	messages := make([]promptMessage, 0, len(prompt))

	for _, message := range prompt {
		encoded, err := encodeMessage(message)
		if err != nil {
			return nil, err
		}

		messages = append(messages, encoded)
	}

	return messages, nil
}

func encodeMessage(message model.Message) (promptMessage, error) {
	switch typed := message.(type) {
	case model.SystemMessage:
		return systemMessage{
			Role:            model.RoleSystem,
			Content:         typed.Content,
			ProviderOptions: typed.ProviderOptions,
		}, nil

	case model.UserMessage:
		parts, err := encodeParts(typed.Content, encodeUserPart)
		if err != nil {
			return nil, err
		}

		return partsMessage{
			Role:            model.RoleUser,
			Content:         parts,
			ProviderOptions: typed.ProviderOptions,
		}, nil

	case model.AssistantMessage:
		parts, err := encodeParts(typed.Content, encodeAssistantPart)
		if err != nil {
			return nil, err
		}

		return partsMessage{
			Role:            model.RoleAssistant,
			Content:         parts,
			ProviderOptions: typed.ProviderOptions,
		}, nil

	case model.ToolMessage:
		parts, err := encodeParts(typed.Content, encodeToolPart)
		if err != nil {
			return nil, err
		}

		return partsMessage{
			Role:            model.RoleTool,
			Content:         parts,
			ProviderOptions: typed.ProviderOptions,
		}, nil

	default:
		return nil, fmt.Errorf("%w: message %T", model.ErrInvalidPrompt, message)
	}
}

// encodeParts converts the parts of a message. Parts that the encoder skips
// return a nil part.
func encodeParts[Part any](parts []Part, encode func(Part) (promptPart, error)) ([]promptPart, error) {
	encoded := make([]promptPart, 0, len(parts))

	for _, part := range parts {
		converted, err := encode(part)
		if err != nil {
			return nil, err
		}

		if converted != nil {
			encoded = append(encoded, converted)
		}
	}

	return encoded, nil
}

func encodeUserPart(part model.UserPart) (promptPart, error) {
	switch typed := part.(type) {
	case model.TextPart:
		return textPart{
			Type:            partText,
			Text:            typed.Text,
			ProviderOptions: typed.ProviderOptions,
		}, nil

	case model.FilePart:
		return encodeFilePart(typed)

	default:
		return nil, fmt.Errorf("%w: user part %T", model.ErrInvalidPrompt, part)
	}
}

// encodeAssistantPart converts an assistant part. Tool approval requests
// are conversation history only and are not sent.
func encodeAssistantPart(part model.AssistantPart) (promptPart, error) {
	switch typed := part.(type) {
	case model.TextPart:
		return textPart{
			Type:            partText,
			Text:            typed.Text,
			ProviderOptions: typed.ProviderOptions,
		}, nil

	case model.FilePart:
		return encodeFilePart(typed)

	case model.ReasoningPart:
		return textPart{
			Type:            partReasoning,
			Text:            typed.Text,
			ProviderOptions: typed.ProviderOptions,
		}, nil

	case model.ReasoningFilePart:
		data, err := encodeFileData(typed.Data)
		if err != nil {
			return nil, err
		}

		return filePart{
			Type:            partReasoningFile,
			Data:            data,
			MediaType:       typed.MediaType,
			ProviderOptions: typed.ProviderOptions,
		}, nil

	case model.CustomPart:
		return customPart{
			Type:            partCustom,
			Kind:            typed.Kind,
			ProviderOptions: typed.ProviderOptions,
		}, nil

	case model.ToolCallPart:
		return toolCallPart{
			Type:             partToolCall,
			ToolCallID:       typed.ToolCallID,
			ToolName:         typed.ToolName,
			Input:            typed.Input,
			ProviderExecuted: typed.ProviderExecuted,
			ProviderOptions:  typed.ProviderOptions,
		}, nil

	case model.ToolResultPart:
		return encodeToolResultPart(typed)

	case model.ToolApprovalRequestPart:
		return nil, nil

	default:
		return nil, fmt.Errorf("%w: assistant part %T", model.ErrInvalidPrompt, part)
	}
}

func encodeToolPart(part model.ToolPart) (promptPart, error) {
	switch typed := part.(type) {
	case model.ToolResultPart:
		return encodeToolResultPart(typed)

	case model.ToolApprovalResponsePart:
		return toolApprovalResponsePart{
			Type:            partToolApprovalResponse,
			ApprovalID:      typed.ApprovalID,
			Approved:        typed.Approved,
			Reason:          typed.Reason,
			ProviderOptions: typed.ProviderOptions,
		}, nil

	default:
		return nil, fmt.Errorf("%w: tool part %T", model.ErrInvalidPrompt, part)
	}
}

func encodeFilePart(part model.FilePart) (promptPart, error) {
	data, err := encodeFileData(part.Data)
	if err != nil {
		return nil, err
	}

	return filePart{
		Type:            partFile,
		Filename:        part.Filename,
		Data:            data,
		MediaType:       part.MediaType,
		ProviderOptions: part.ProviderOptions,
	}, nil
}

func encodeToolResultPart(part model.ToolResultPart) (promptPart, error) {
	output, err := encodeToolResultOutput(part.Output)
	if err != nil {
		return nil, err
	}

	return toolResultPart{
		Type:            partToolResult,
		ToolCallID:      part.ToolCallID,
		ToolName:        part.ToolName,
		Output:          output,
		ProviderOptions: part.ProviderOptions,
	}, nil
}

func encodeToolResultOutput(output model.ToolResultOutput) (toolResultOutput, error) {
	switch typed := output.(type) {
	case model.TextOutput:
		return stringOutput(outputText, typed.Value, typed.ProviderOptions)

	case model.ErrorTextOutput:
		return stringOutput(outputErrorText, typed.Value, typed.ProviderOptions)

	case model.JSONOutput:
		return toolResultOutput{
			Type:            outputJSON,
			Value:           jsonOrNull(typed.Value),
			ProviderOptions: typed.ProviderOptions,
		}, nil

	case model.ErrorJSONOutput:
		return toolResultOutput{
			Type:            outputErrorJSON,
			Value:           jsonOrNull(typed.Value),
			ProviderOptions: typed.ProviderOptions,
		}, nil

	case model.ExecutionDeniedOutput:
		return toolResultOutput{
			Type:            outputExecutionDenied,
			Reason:          typed.Reason,
			ProviderOptions: typed.ProviderOptions,
		}, nil

	case model.ContentOutput:
		return contentOutput(typed)

	default:
		return toolResultOutput{}, fmt.Errorf("%w: tool result output %T", model.ErrInvalidPrompt, output)
	}
}

func stringOutput(outputType string, value string, options model.ProviderOptions) (toolResultOutput, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return toolResultOutput{}, fmt.Errorf("encode tool result: %w", err)
	}

	return toolResultOutput{
		Type:            outputType,
		Value:           encoded,
		ProviderOptions: options,
	}, nil
}

// jsonOrNull returns raw, or JSON null when it is absent.
func jsonOrNull(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return json.RawMessage("null")
	}

	return raw
}

func contentOutput(output model.ContentOutput) (toolResultOutput, error) {
	items := make([]outputItem, 0, len(output.Value))

	for _, item := range output.Value {
		switch typed := item.(type) {
		case model.ToolResultText:
			items = append(items, outputTextItem{
				Type:            partText,
				Text:            typed.Text,
				ProviderOptions: typed.ProviderOptions,
			})

		case model.ToolResultFile:
			data, err := encodeFileData(typed.Data)
			if err != nil {
				return toolResultOutput{}, err
			}

			items = append(items, outputFileItem{
				Type:            partFile,
				Data:            data,
				MediaType:       typed.MediaType,
				Filename:        typed.Filename,
				ProviderOptions: typed.ProviderOptions,
			})

		case model.ToolResultCustom:
			items = append(items, outputCustomItem{
				Type:            partCustom,
				ProviderOptions: typed.ProviderOptions,
			})

		default:
			return toolResultOutput{}, fmt.Errorf("%w: tool result content %T", model.ErrInvalidPrompt, item)
		}
	}

	encoded, err := json.Marshal(items)
	if err != nil {
		return toolResultOutput{}, fmt.Errorf("encode tool result content: %w", err)
	}

	return toolResultOutput{
		Type:  outputContent,
		Value: encoded,
	}, nil
}
