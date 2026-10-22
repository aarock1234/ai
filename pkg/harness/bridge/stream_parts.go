package bridge

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/model"
)

// wireStreamPart holds every field of the stream part frames. Required
// fields are pointers, so that their absence is detected.
type wireStreamPart struct {
	ID              *string      `json:"id"`
	Delta           *string      `json:"delta"`
	HarnessMetadata metadataJSON `json:"harnessMetadata"`

	ToolName         *string      `json:"toolName"`
	ToolCallID       *string      `json:"toolCallId"`
	ApprovalID       *string      `json:"approvalId"`
	Input            *string      `json:"input"`
	ProviderExecuted bool         `json:"providerExecuted"`
	Dynamic          bool         `json:"dynamic"`
	Title            string       `json:"title"`
	NativeName       string       `json:"nativeName"`
	ProviderMetadata metadataJSON `json:"providerMetadata"`

	Result      json.RawMessage `json:"result"`
	IsError     bool            `json:"isError"`
	Preliminary bool            `json:"preliminary"`

	FinishReason *wireFinishReason `json:"finishReason"`
	Usage        *wireUsage        `json:"usage"`
	TotalUsage   *wireUsage        `json:"totalUsage"`

	Event        *harness.FileChangeEvent   `json:"event"`
	Path         *string                    `json:"path"`
	Trigger      *harness.CompactionTrigger `json:"trigger"`
	Summary      *string                    `json:"summary"`
	TokensBefore *int                       `json:"tokensBefore"`
	TokensAfter  *int                       `json:"tokensAfter"`

	Warnings []wireWarning   `json:"warnings"`
	ModelID  string          `json:"modelId"`
	Error    json.RawMessage `json:"error"`
	RawValue json.RawMessage `json:"rawValue"`
}

// wireFinishReason is the wire form of model.FinishReason.
type wireFinishReason struct {
	Unified *model.UnifiedFinishReason `json:"unified"`
	Raw     string                     `json:"raw"`
}

// wireUsage is the wire form of model.Usage.
type wireUsage struct {
	InputTokens *struct {
		Total      *int `json:"total"`
		NoCache    *int `json:"noCache"`
		CacheRead  *int `json:"cacheRead"`
		CacheWrite *int `json:"cacheWrite"`
	} `json:"inputTokens"`
	OutputTokens *struct {
		Total     *int `json:"total"`
		Text      *int `json:"text"`
		Reasoning *int `json:"reasoning"`
	} `json:"outputTokens"`
	Raw json.RawMessage `json:"raw"`
}

// wireWarning is the wire form of a harness call warning.
type wireWarning struct {
	Type    string `json:"type"`
	Setting string `json:"setting"`
	Tool    string `json:"tool"`
	Details string `json:"details"`
	Message string `json:"message"`
}

// decodeStreamPart decodes and validates a stream part frame.
func decodeStreamPart(frameType string, data []byte) (harness.StreamPart, error) {
	var wire wireStreamPart
	if err := decodeWire(data, &wire); err != nil {
		return nil, err
	}

	decode, ok := streamPartDecoders[frameType]
	if !ok {
		return nil, fmt.Errorf("%w: unknown frame type %q", ErrInvalidFrame, frameType)
	}

	part, err := decode(wire)
	if err != nil {
		return nil, fmt.Errorf("%w: %s frame: %w", ErrInvalidFrame, frameType, err)
	}

	return part, nil
}

// streamPartDecoders decode the stream part frames by type.
var streamPartDecoders = map[string]func(wire wireStreamPart) (harness.StreamPart, error){
	"stream-start": func(wire wireStreamPart) (harness.StreamPart, error) {
		return harness.StreamStartPart{
			Warnings: warningsOf(wire.Warnings),
			ModelID:  wire.ModelID,
		}, nil
	},
	"text-start": func(wire wireStreamPart) (harness.StreamPart, error) {
		id, err := require(wire.ID, "id")

		return harness.TextStartPart{
			ID:              id,
			HarnessMetadata: toMetadata(wire.HarnessMetadata),
		}, err
	},
	"text-delta": func(wire wireStreamPart) (harness.StreamPart, error) {
		id, delta, err := requireBlockDelta(wire)

		return harness.TextDeltaPart{
			ID:              id,
			Delta:           delta,
			HarnessMetadata: toMetadata(wire.HarnessMetadata),
		}, err
	},
	"text-end": func(wire wireStreamPart) (harness.StreamPart, error) {
		id, err := require(wire.ID, "id")

		return harness.TextEndPart{
			ID:              id,
			HarnessMetadata: toMetadata(wire.HarnessMetadata),
		}, err
	},
	"reasoning-start": func(wire wireStreamPart) (harness.StreamPart, error) {
		id, err := require(wire.ID, "id")

		return harness.ReasoningStartPart{
			ID:              id,
			HarnessMetadata: toMetadata(wire.HarnessMetadata),
		}, err
	},
	"reasoning-delta": func(wire wireStreamPart) (harness.StreamPart, error) {
		id, delta, err := requireBlockDelta(wire)

		return harness.ReasoningDeltaPart{
			ID:              id,
			Delta:           delta,
			HarnessMetadata: toMetadata(wire.HarnessMetadata),
		}, err
	},
	"reasoning-end": func(wire wireStreamPart) (harness.StreamPart, error) {
		id, err := require(wire.ID, "id")

		return harness.ReasoningEndPart{
			ID:              id,
			HarnessMetadata: toMetadata(wire.HarnessMetadata),
		}, err
	},
	"tool-input-start":      decodeToolInputStart,
	"tool-input-delta":      decodeToolInputDelta,
	"tool-input-end":        decodeToolInputEnd,
	"tool-call":             decodeToolCall,
	"tool-approval-request": decodeToolApprovalRequest,
	"tool-result":           decodeToolResult,
	"finish-step":           decodeFinishStep,
	"finish":                decodeFinish,
	"file-change":           decodeFileChange,
	"compaction":            decodeCompaction,
	"error": func(wire wireStreamPart) (harness.StreamPart, error) {
		return harness.ErrorPart{
			Err: &RemoteError{
				Value: wire.Error,
			},
		}, nil
	},
	"raw": func(wire wireStreamPart) (harness.StreamPart, error) {
		return harness.RawPart{
			Value: wire.RawValue,
		}, nil
	},
}

// decodeToolInputStart decodes a tool-input-start frame.
func decodeToolInputStart(wire wireStreamPart) (harness.StreamPart, error) {
	id, err := require(wire.ID, "id")
	if err != nil {
		return nil, err
	}

	toolName, err := require(wire.ToolName, "toolName")
	if err != nil {
		return nil, err
	}

	part := harness.ToolInputStartPart{
		ID:               id,
		ToolName:         toolName,
		ProviderExecuted: wire.ProviderExecuted,
		Dynamic:          wire.Dynamic,
		Title:            wire.Title,
		ProviderMetadata: toProviderMetadata(wire.ProviderMetadata),
	}

	return part, nil
}

// decodeToolInputDelta decodes a tool-input-delta frame.
func decodeToolInputDelta(wire wireStreamPart) (harness.StreamPart, error) {
	id, delta, err := requireBlockDelta(wire)
	if err != nil {
		return nil, err
	}

	part := harness.ToolInputDeltaPart{
		ID:               id,
		Delta:            delta,
		ProviderMetadata: toProviderMetadata(wire.ProviderMetadata),
	}

	return part, nil
}

// decodeToolInputEnd decodes a tool-input-end frame.
func decodeToolInputEnd(wire wireStreamPart) (harness.StreamPart, error) {
	id, err := require(wire.ID, "id")
	if err != nil {
		return nil, err
	}

	part := harness.ToolInputEndPart{
		ID:               id,
		ProviderMetadata: toProviderMetadata(wire.ProviderMetadata),
	}

	return part, nil
}

// decodeToolCall decodes a tool-call frame.
func decodeToolCall(wire wireStreamPart) (harness.StreamPart, error) {
	toolCallID, toolName, err := requireToolIdentity(wire)
	if err != nil {
		return nil, err
	}

	input, err := require(wire.Input, "input")
	if err != nil {
		return nil, err
	}

	part := harness.ToolCallPart{
		NativeName: wire.NativeName,

		ToolCallID:       toolCallID,
		ToolName:         toolName,
		Input:            input,
		ProviderExecuted: wire.ProviderExecuted,
		Dynamic:          wire.Dynamic,
		ProviderMetadata: toProviderMetadata(wire.ProviderMetadata),
	}

	return part, nil
}

// decodeToolApprovalRequest decodes a tool-approval-request frame.
func decodeToolApprovalRequest(wire wireStreamPart) (harness.StreamPart, error) {
	approvalID, err := require(wire.ApprovalID, "approvalId")
	if err != nil {
		return nil, err
	}

	toolCallID, err := require(wire.ToolCallID, "toolCallId")
	if err != nil {
		return nil, err
	}

	part := harness.ToolApprovalRequestPart{
		ApprovalID:       approvalID,
		ToolCallID:       toolCallID,
		ProviderMetadata: toProviderMetadata(wire.ProviderMetadata),
	}

	return part, nil
}

// decodeToolResult decodes a tool-result frame. A missing result is kept as
// JSON null, as upstream accepts it.
func decodeToolResult(wire wireStreamPart) (harness.StreamPart, error) {
	toolCallID, toolName, err := requireToolIdentity(wire)
	if err != nil {
		return nil, err
	}

	result := wire.Result
	if len(result) == 0 {
		result = json.RawMessage("null")
	}

	part := harness.ToolResultPart{
		ToolCallID:       toolCallID,
		ToolName:         toolName,
		Result:           result,
		IsError:          wire.IsError,
		Preliminary:      wire.Preliminary,
		Dynamic:          wire.Dynamic,
		ProviderMetadata: toProviderMetadata(wire.ProviderMetadata),
	}

	return part, nil
}

// decodeFinishStep decodes a finish-step frame.
func decodeFinishStep(wire wireStreamPart) (harness.StreamPart, error) {
	finishReason, err := finishReasonOf(wire.FinishReason)
	if err != nil {
		return nil, err
	}

	usage, err := usageOf(wire.Usage, "usage")
	if err != nil {
		return nil, err
	}

	return harness.FinishStepPart{
		FinishReason:    finishReason,
		Usage:           usage,
		HarnessMetadata: toMetadata(wire.HarnessMetadata),
	}, nil
}

// decodeFinish decodes a finish frame.
func decodeFinish(wire wireStreamPart) (harness.StreamPart, error) {
	finishReason, err := finishReasonOf(wire.FinishReason)
	if err != nil {
		return nil, err
	}

	totalUsage, err := usageOf(wire.TotalUsage, "totalUsage")
	if err != nil {
		return nil, err
	}

	return harness.FinishPart{
		FinishReason:    finishReason,
		TotalUsage:      totalUsage,
		HarnessMetadata: toMetadata(wire.HarnessMetadata),
	}, nil
}

// decodeFileChange decodes a file-change frame.
func decodeFileChange(wire wireStreamPart) (harness.StreamPart, error) {
	if wire.Event == nil || !isFileChangeEvent(*wire.Event) {
		return nil, fmt.Errorf("event must be create, modify or delete")
	}

	filePath, err := require(wire.Path, "path")
	if err != nil {
		return nil, err
	}

	return harness.FileChangePart{
		Event:           *wire.Event,
		Path:            filePath,
		HarnessMetadata: toMetadata(wire.HarnessMetadata),
	}, nil
}

// decodeCompaction decodes a compaction frame.
func decodeCompaction(wire wireStreamPart) (harness.StreamPart, error) {
	if wire.Trigger == nil || (*wire.Trigger != harness.CompactionManual && *wire.Trigger != harness.CompactionAuto) {
		return nil, fmt.Errorf("trigger must be manual or auto")
	}

	summary, err := require(wire.Summary, "summary")
	if err != nil {
		return nil, err
	}

	return harness.CompactionPart{
		Trigger:         *wire.Trigger,
		Summary:         summary,
		TokensBefore:    wire.TokensBefore,
		TokensAfter:     wire.TokensAfter,
		HarnessMetadata: toMetadata(wire.HarnessMetadata),
	}, nil
}

// finishReasonOf converts a required finish reason.
func finishReasonOf(wire *wireFinishReason) (model.FinishReason, error) {
	if wire == nil || wire.Unified == nil || !isUnifiedFinishReason(*wire.Unified) {
		return model.FinishReason{}, fmt.Errorf("finishReason requires a known unified reason")
	}

	return model.FinishReason{
		Unified: *wire.Unified,
		Raw:     wire.Raw,
	}, nil
}

// usageOf converts required usage.
func usageOf(wire *wireUsage, field string) (model.Usage, error) {
	if wire == nil || wire.InputTokens == nil || wire.OutputTokens == nil {
		return model.Usage{}, fmt.Errorf("%s requires inputTokens and outputTokens", field)
	}

	return model.Usage{
		InputTokens: model.InputTokens{
			Total:      wire.InputTokens.Total,
			NoCache:    wire.InputTokens.NoCache,
			CacheRead:  wire.InputTokens.CacheRead,
			CacheWrite: wire.InputTokens.CacheWrite,
		},
		OutputTokens: model.OutputTokens{
			Total:     wire.OutputTokens.Total,
			Text:      wire.OutputTokens.Text,
			Reasoning: wire.OutputTokens.Reasoning,
		},
		Raw: wire.Raw,
	}, nil
}

// warningsOf converts harness call warnings into model warnings.
func warningsOf(warnings []wireWarning) []model.Warning {
	if len(warnings) == 0 {
		return nil
	}

	converted := make([]model.Warning, 0, len(warnings))

	for _, warning := range warnings {
		switch warning.Type {
		case "unsupported-setting":
			converted = append(converted, model.Warning{
				Type:    model.WarningUnsupported,
				Setting: warning.Setting,
				Details: warning.Details,
			})
		case "unsupported-tool":
			converted = append(converted, model.Warning{
				Type:    model.WarningUnsupported,
				Feature: warning.Tool,
				Details: warning.Details,
			})
		default:
			converted = append(converted, model.Warning{
				Type:    model.WarningOther,
				Message: warning.Message,
			})
		}
	}

	return converted
}

// require returns a required string field.
func require(value *string, field string) (string, error) {
	if value == nil {
		return "", fmt.Errorf("%s is required", field)
	}

	return *value, nil
}

// requireBlockDelta returns the ID and delta of a delta frame.
func requireBlockDelta(wire wireStreamPart) (string, string, error) {
	id, err := require(wire.ID, "id")
	if err != nil {
		return "", "", err
	}

	delta, err := require(wire.Delta, "delta")

	return id, delta, err
}

// requireToolIdentity returns the tool call ID and tool name of a frame.
func requireToolIdentity(wire wireStreamPart) (string, string, error) {
	toolCallID, err := require(wire.ToolCallID, "toolCallId")
	if err != nil {
		return "", "", err
	}

	toolName, err := require(wire.ToolName, "toolName")

	return toolCallID, toolName, err
}

// isFileChangeEvent reports a known file change event.
func isFileChangeEvent(event harness.FileChangeEvent) bool {
	return event == harness.FileCreated || event == harness.FileModified || event == harness.FileDeleted
}

// isUnifiedFinishReason reports a known unified finish reason.
func isUnifiedFinishReason(reason model.UnifiedFinishReason) bool {
	switch reason {
	case model.FinishStop, model.FinishLength, model.FinishContentFilter, model.FinishToolCalls, model.FinishError, model.FinishOther:
		return true
	default:
		return false
	}
}
