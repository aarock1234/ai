package ui

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// chunkConverter converts single StreamText parts into UI message chunks,
// the upstream toUIMessageChunk.
type chunkConverter struct {
	tools []ai.Tool

	sendReasoning bool
	sendSources   bool
	sendStart     bool
	sendFinish    bool

	onError func(err error) string

	responseMessageID string
}

// convert returns the chunk of a part. It reports false for parts without a
// chunk. metadata is the message metadata for start and finish chunks.
func (c chunkConverter) convert(part ai.StreamPart, metadata json.RawMessage) (Chunk, bool, error) {
	switch typed := part.(type) {
	case ai.TextStartPart:
		return TextStartChunk{
			ID:               typed.ID,
			ProviderMetadata: typed.ProviderMetadata,
		}, true, nil

	case ai.TextDeltaPart:
		return TextDeltaChunk{
			ID:               typed.ID,
			Delta:            typed.Text,
			ProviderMetadata: typed.ProviderMetadata,
		}, true, nil

	case ai.TextEndPart:
		return TextEndChunk{
			ID:               typed.ID,
			Citations:        citationsFromModel(typed.Citations),
			ProviderMetadata: typed.ProviderMetadata,
		}, true, nil

	case ai.ReasoningStartPart:
		return ReasoningStartChunk{
			ID:               typed.ID,
			ProviderMetadata: typed.ProviderMetadata,
		}, c.sendReasoning, nil

	case ai.ReasoningDeltaPart:
		return ReasoningDeltaChunk{
			ID:               typed.ID,
			Delta:            typed.Text,
			ProviderMetadata: typed.ProviderMetadata,
		}, c.sendReasoning, nil

	case ai.ReasoningEndPart:
		return ReasoningEndChunk{
			ID:               typed.ID,
			ProviderMetadata: typed.ProviderMetadata,
		}, c.sendReasoning, nil

	case ai.FileContent:
		return FileChunk{
			URL:              dataURL(typed.File),
			MediaType:        typed.File.MediaType,
			ProviderMetadata: typed.ProviderMetadata,
		}, true, nil

	case ai.ReasoningFileContent:
		return ReasoningFileChunk{
			URL:              dataURL(typed.File),
			MediaType:        typed.File.MediaType,
			ProviderMetadata: typed.ProviderMetadata,
		}, c.sendReasoning, nil

	case ai.SourceContent:
		return c.sourceChunk(typed)

	case ai.CustomContent:
		return CustomChunk{
			Kind:             typed.Kind,
			ProviderMetadata: typed.ProviderMetadata,
		}, true, nil

	case ai.StartStepPart:
		return StartStepChunk{}, true, nil

	case ai.FinishStepPart:
		return FinishStepChunk{}, true, nil

	case ai.StartPart:
		return StartChunk{
			MessageID:       c.responseMessageID,
			MessageMetadata: metadata,
		}, c.sendStart, nil

	case ai.FinishPart:
		return FinishChunk{
			FinishReason:    typed.FinishReason,
			MessageMetadata: metadata,
		}, c.sendFinish, nil

	case ai.AbortPart:
		return AbortChunk{
			Reason: typed.Reason,
		}, true, nil

	case ai.ErrorPart:
		return ErrorChunk{
			ErrorText: c.onError(typed.Err),
		}, true, nil

	case ai.ToolInputEndPart, ai.RawPart:
		return nil, false, nil

	default:
		return c.toolChunk(part)
	}
}

// toolChunk converts the parts of tool calls.
func (c chunkConverter) toolChunk(part ai.StreamPart) (Chunk, bool, error) {
	switch typed := part.(type) {
	case ai.ToolInputStartPart:
		return ToolInputStartChunk{
			ToolCallID:       typed.ID,
			ToolName:         typed.ToolName,
			InputEncoding:    typed.InputEncoding,
			ProviderExecuted: typed.ProviderExecuted,
			ProviderMetadata: typed.ProviderMetadata,
			ToolMetadata:     typed.ToolMetadata,
			Dynamic:          c.isDynamic(typed.ToolName, typed.Dynamic),
			Title:            typed.Title,
		}, true, nil

	case ai.ToolInputDeltaPart:
		return ToolInputDeltaChunk{
			ToolCallID:     typed.ID,
			InputTextDelta: typed.Delta,
		}, true, nil

	case ai.ToolCall:
		return c.toolCallChunk(typed), true, nil

	case ai.ToolApprovalRequest:
		return ToolApprovalRequestChunk{
			ApprovalID:       typed.ApprovalID,
			ToolCallID:       typed.ToolCall.ToolCallID,
			InputSchemaInput: typed.ToolCall.InputSchemaInput(),
			Reason:           typed.Reason,
			IsAutomatic:      typed.IsAutomatic,
			Signature:        typed.Signature,
		}, true, nil

	case ai.ToolApprovalResponse:
		return ToolApprovalResponseChunk{
			ApprovalID:       typed.ApprovalID,
			Approved:         typed.Approved,
			Reason:           typed.Reason,
			ProviderExecuted: typed.ProviderExecuted,
		}, true, nil

	case ai.ToolResult:
		return c.toolResultChunk(typed)

	case ai.ToolError:
		return ToolOutputErrorChunk{
			ToolCallID:       typed.ToolCallID,
			ErrorText:        c.toolErrorText(typed),
			ProviderExecuted: typed.ProviderExecuted,
			ProviderMetadata: typed.ProviderMetadata,
			ToolMetadata:     typed.ToolMetadata,
			Dynamic:          c.isDynamic(typed.ToolName, typed.Dynamic),
		}, true, nil

	case ai.ToolOutputDeniedPart:
		return ToolOutputDeniedChunk{
			ToolCallID: typed.ToolCallID,
		}, true, nil

	default:
		return nil, false, fmt.Errorf("unsupported stream part %T", part)
	}
}

func (c chunkConverter) toolCallChunk(call ai.ToolCall) Chunk {
	dynamic := c.isDynamic(call.ToolName, call.Dynamic)

	if call.Invalid {
		return ToolInputErrorChunk{
			ToolCallID:       call.ToolCallID,
			ToolName:         call.ToolName,
			Input:            call.Input,
			ProviderExecuted: call.ProviderExecuted,
			ProviderMetadata: call.ProviderMetadata,
			ToolMetadata:     call.ToolMetadata,
			Dynamic:          dynamic,
			ErrorText:        c.onError(call.Error),
			Title:            call.Title,
		}
	}

	return ToolInputAvailableChunk{
		ToolCallID:       call.ToolCallID,
		ToolName:         call.ToolName,
		Input:            call.Input,
		ProviderExecuted: call.ProviderExecuted,
		ProviderMetadata: call.ProviderMetadata,
		ToolMetadata:     call.ToolMetadata,
		Dynamic:          dynamic,
		Title:            call.Title,
	}
}

func (c chunkConverter) toolResultChunk(result ai.ToolResult) (Chunk, bool, error) {
	output, err := encodeValue(result.Output)
	if err != nil {
		return nil, false, fmt.Errorf("encode output of tool call %s: %w", result.ToolCallID, err)
	}

	return ToolOutputAvailableChunk{
		ToolCallID:       result.ToolCallID,
		Output:           output,
		ProviderExecuted: result.ProviderExecuted,
		ProviderMetadata: result.ProviderMetadata,
		ToolMetadata:     result.ToolMetadata,
		Dynamic:          c.isDynamic(result.ToolName, result.Dynamic),
		Preliminary:      result.Preliminary,
	}, true, nil
}

// toolErrorText formats an application tool error with onError. Errors of
// provider-executed tools keep the provider's data so that they survive a
// round trip through model messages: JSON strings are sent as their text and
// other JSON values as JSON.
func (c chunkConverter) toolErrorText(toolError ai.ToolError) string {
	if !toolError.ProviderExecuted {
		return c.onError(toolError.Error)
	}

	if providerErr, ok := errors.AsType[*ai.ProviderToolError](toolError.Error); ok {
		var text string
		if err := json.Unmarshal(providerErr.Result, &text); err == nil {
			return text
		}

		return string(providerErr.Result)
	}

	if toolError.Error == nil {
		return ""
	}

	return toolError.Error.Error()
}

func (c chunkConverter) sourceChunk(source ai.SourceContent) (Chunk, bool, error) {
	if !c.sendSources {
		return nil, false, nil
	}

	switch source.SourceType {
	case model.SourceURL:
		return SourceURLChunk{
			SourceID:         source.ID,
			URL:              source.URL,
			Title:            source.Title,
			ProviderMetadata: source.ProviderMetadata,
		}, true, nil

	case model.SourceDocument:
		return SourceDocumentChunk{
			SourceID:         source.ID,
			MediaType:        source.MediaType,
			Title:            source.Title,
			Filename:         source.Filename,
			ProviderMetadata: source.ProviderMetadata,
		}, true, nil

	default:
		return nil, false, nil
	}
}

// isDynamic reports whether a tool chunk is marked as dynamic: for declared
// tools when the tool is dynamic, and otherwise, as for provider-executed
// tools, when the part says so.
func (c chunkConverter) isDynamic(toolName string, partDynamic bool) bool {
	for _, tool := range c.tools {
		if tool.Name == toolName {
			return tool.Kind == ai.ToolKindDynamic
		}
	}

	return partDynamic
}

// dataURL encodes a generated file as a base64 data URL.
func dataURL(file ai.GeneratedFile) string {
	return "data:" + file.MediaType + ";base64," + file.Base64()
}
