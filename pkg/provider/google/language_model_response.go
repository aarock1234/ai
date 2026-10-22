package google

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/toolnames"
	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/model"
)

// Finish reasons with special handling.
const (
	finishReasonStop         = "STOP"
	finishReasonMaxTokens    = "MAX_TOKENS"
	finishReasonContinuation = "CONTINUATION"
)

// serverToolNamePrefix prefixes the dynamic tool names of server-side tool
// calls, such as "server:GOOGLE_SEARCH_WEB".
const serverToolNamePrefix = "server:"

// mapFinishReason maps a Gemini finish reason. A stop with client tool calls
// becomes a tool-calls finish.
func mapFinishReason(raw string, hasToolCalls bool) model.UnifiedFinishReason {
	switch raw {
	case finishReasonStop:
		if hasToolCalls {
			return model.FinishToolCalls
		}

		return model.FinishStop

	case finishReasonMaxTokens, finishReasonContinuation:
		return model.FinishLength

	case "IMAGE_SAFETY", "RECITATION", "SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII":
		return model.FinishContentFilter

	case "MALFORMED_FUNCTION_CALL":
		return model.FinishError

	default:
		return model.FinishOther
	}
}

// isConfirmedBlockReason reports a prompt block reason other than the
// unspecified values.
func isConfirmedBlockReason(reason *string) bool {
	if reason == nil {
		return false
	}

	switch *reason {
	case "", "BLOCK_REASON_UNSPECIFIED", "BLOCKED_REASON_UNSPECIFIED":
		return false
	default:
		return true
	}
}

// contentConverter converts the parts of one generateContent response.
// Code execution results and server-side tool responses belong to the call
// that precedes them.
type contentConverter struct {
	toolNames            toolnames.Mapping
	jsonResponseToolName string
	generateID           func() string

	// metadataNamespaces key the part metadata.
	metadataNamespaces []string

	lastCodeExecutionCallID string
	lastServerToolCallID    string
}

func (c *contentConverter) convert(parts []part) ([]model.Content, error) {
	var contents []model.Content

	for _, responsePart := range parts {
		converted, err := c.convertPart(responsePart, contents)
		if err != nil {
			return nil, err
		}

		contents = converted
	}

	return contents, nil
}

// convertPart appends the content of one part. An empty text part with a
// thought signature attaches the signature to the previous content.
func (c *contentConverter) convertPart(responsePart part, contents []model.Content) ([]model.Content, error) {
	metadata := signatureMetadata(c.metadataNamespaces, responsePart.ThoughtSignature)

	switch {
	case responsePart.ExecutableCode != nil && responsePart.ExecutableCode.Code != "":
		return append(contents, c.codeExecutionCall(*responsePart.ExecutableCode)), nil

	case responsePart.CodeExecutionResult != nil:
		result, ok := c.codeExecutionResult(*responsePart.CodeExecutionResult)
		if !ok {
			return contents, nil
		}

		return append(contents, result), nil

	case responsePart.Text != nil:
		return c.appendText(responsePart, contents), nil

	case responsePart.FunctionCall != nil && responsePart.FunctionCall.Name != "":
		return append(contents, c.functionCall(*responsePart.FunctionCall, metadata)), nil

	case responsePart.InlineData != nil:
		file, err := inlineFile(responsePart, metadata)
		if err != nil {
			return nil, err
		}

		return append(contents, file), nil

	case responsePart.ToolCall != nil:
		return append(contents, c.serverToolCall(*responsePart.ToolCall, responsePart.ThoughtSignature)), nil

	case responsePart.ToolResponse != nil:
		return append(contents, c.serverToolResult(*responsePart.ToolResponse, responsePart.ThoughtSignature)), nil

	default:
		return contents, nil
	}
}

func (c *contentConverter) appendText(responsePart part, contents []model.Content) []model.Content {
	if c.jsonResponseToolName != "" && !responsePart.Thought {
		return contents
	}

	metadata := signatureMetadata(c.metadataNamespaces, responsePart.ThoughtSignature)
	text := *responsePart.Text

	if text == "" {
		if metadata != nil && len(contents) > 0 {
			contents[len(contents)-1] = withProviderMetadata(contents[len(contents)-1], metadata)
		}

		return contents
	}

	if responsePart.Thought {
		return append(contents, model.Reasoning{
			Text:             text,
			ProviderMetadata: metadata,
		})
	}

	return append(contents, model.Text{
		Text:             text,
		ProviderMetadata: metadata,
	})
}

func (c *contentConverter) codeExecutionCall(input CodeExecutionInput) model.ToolCall {
	callID := c.generateID()
	c.lastCodeExecutionCallID = callID

	return model.ToolCall{
		ToolCallID:       callID,
		ToolName:         c.toolNames.ToApplication(codeExecutionToolName),
		Input:            encodeCompact(input),
		ProviderExecuted: true,
	}
}

// codeExecutionResult reports false for a result without a preceding call,
// which cannot be attributed.
func (c *contentConverter) codeExecutionResult(output CodeExecutionOutput) (model.ToolResult, bool) {
	if c.lastCodeExecutionCallID == "" {
		return model.ToolResult{}, false
	}

	return model.ToolResult{
		ToolCallID: c.lastCodeExecutionCallID,
		ToolName:   c.toolNames.ToApplication(codeExecutionToolName),
		Result:     json.RawMessage(encodeCompact(output)),
	}, true
}

// functionCall converts a client function call. The JSON response tool
// becomes text holding the structured output.
func (c *contentConverter) functionCall(call functionCall, metadata model.ProviderMetadata) model.Content {
	if call.Name == c.jsonResponseToolName {
		return model.Text{
			Text:             functionCallArgumentsText(call.Args),
			ProviderMetadata: metadata,
		}
	}

	callID := call.ID
	if callID == "" {
		callID = c.generateID()
	}

	return model.ToolCall{
		ToolCallID:       callID,
		ToolName:         call.Name,
		Input:            compactArguments(call.Args),
		ProviderMetadata: metadata,
	}
}

func (c *contentConverter) serverToolCall(call serverToolCall, signature string) model.ToolCall {
	callID := call.ID
	if callID == "" {
		callID = c.generateID()
	}

	c.lastServerToolCallID = callID

	return model.ToolCall{
		ToolCallID:       callID,
		ToolName:         serverToolNamePrefix + call.ToolType,
		Input:            compactArguments(call.Args),
		ProviderExecuted: true,
		Dynamic:          true,
		ProviderMetadata: serverToolMetadata(c.metadataNamespaces, signature, callID, call.ToolType),
	}
}

func (c *contentConverter) serverToolResult(response serverToolResponse, signature string) model.ToolResult {
	callID := firstNonEmpty(c.lastServerToolCallID, response.ID)
	if callID == "" {
		callID = c.generateID()
	}

	c.lastServerToolCallID = ""

	return model.ToolResult{
		ToolCallID:       callID,
		ToolName:         serverToolNamePrefix + response.ToolType,
		Result:           json.RawMessage(compactArguments(response.Response)),
		ProviderMetadata: serverToolMetadata(c.metadataNamespaces, signature, callID, response.ToolType),
	}
}

// inlineFile converts generated inline data. Thought files are reasoning
// files.
func inlineFile(responsePart part, metadata model.ProviderMetadata) (model.Content, error) {
	data, err := base64.StdEncoding.DecodeString(responsePart.InlineData.Data)
	if err != nil {
		return nil, fmt.Errorf("%w: decode inline data: %w", model.ErrInvalidResponse, err)
	}

	fileData := model.FileBytes{
		Data: data,
	}

	if responsePart.Thought {
		return model.ReasoningFile{
			MediaType:        responsePart.InlineData.MimeType,
			Data:             fileData,
			ProviderMetadata: metadata,
		}, nil
	}

	return model.File{
		MediaType:        responsePart.InlineData.MimeType,
		Data:             fileData,
		ProviderMetadata: metadata,
	}, nil
}

// signatureMetadata returns the metadata of a part with a thought signature,
// or nil.
func signatureMetadata(namespaces []string, signature string) model.ProviderMetadata {
	if signature == "" {
		return nil
	}

	return wrapMetadata(namespaces, PartMetadata{
		ThoughtSignature: signature,
	})
}

func serverToolMetadata(namespaces []string, signature string, callID string, toolType string) model.ProviderMetadata {
	return wrapMetadata(namespaces, PartMetadata{
		ThoughtSignature: signature,
		ServerToolCallID: callID,
		ServerToolType:   toolType,
	})
}

// withProviderMetadata replaces the metadata of a content item.
func withProviderMetadata(item model.Content, metadata model.ProviderMetadata) model.Content {
	switch item := item.(type) {
	case model.Text:
		item.ProviderMetadata = metadata

		return item
	case model.Reasoning:
		item.ProviderMetadata = metadata

		return item
	case model.File:
		item.ProviderMetadata = metadata

		return item
	case model.ReasoningFile:
		item.ProviderMetadata = metadata

		return item
	case model.ToolCall:
		item.ProviderMetadata = metadata

		return item
	case model.ToolResult:
		item.ProviderMetadata = metadata

		return item
	case model.Source:
		item.ProviderMetadata = metadata

		return item
	default:
		return item
	}
}

// compactArguments returns JSON arguments as compact text, with "{}" for
// missing arguments.
func compactArguments(arguments json.RawMessage) string {
	if !wirejson.HasValue(arguments) {
		return "{}"
	}

	var compacted bytes.Buffer
	if err := json.Compact(&compacted, arguments); err != nil {
		return string(arguments)
	}

	return compacted.String()
}

// functionCallArgumentsText returns arguments sent as a JSON string as that
// string, and other arguments as JSON.
func functionCallArgumentsText(arguments json.RawMessage) string {
	var text string
	if err := json.Unmarshal(arguments, &text); err == nil {
		return text
	}

	return compactArguments(arguments)
}

func encodeCompact[T any](value T) string {
	// The encoded values are plain structs, which always encode.
	encoded, _ := json.Marshal(value)

	return string(encoded)
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}

	return ""
}
