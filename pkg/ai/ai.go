// Package ai provides the high-level functions for interacting with
// language models through the AI SDK. Consumers import this package
// alongside a provider package (e.g. pkg/provider/openai) to generate
// text, stream responses, produce structured output, and embed text.
package ai

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// GenerateText sends a non-streaming generation request to the model
// and returns the complete result. When tools are provided and
// WithMaxSteps is greater than 1, GenerateText runs the automatic
// tool loop: it calls tools, feeds results back to the model, and
// repeats until the model stops or the step limit is reached.
func GenerateText(ctx context.Context, m model.Model, opts ...Option) (*GenerateResult, error) {
	cfg := resolveConfig(opts)

	if cfg.timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, cfg.timeout)
		defer cancel()
	}

	req := buildRequest(cfg)
	toolMap := buildToolMap(cfg.tools)

	var steps []Step
	var totalUsage Usage

	for step := range cfg.maxSteps {
		resp, err := m.DoGenerate(ctx, req)
		if err != nil {
			return nil, fmt.Errorf("step %d: %w", step, err)
		}

		stepResult := extractStep(resp)
		steps = append(steps, stepResult)
		totalUsage.InputTokens += stepResult.Usage.InputTokens
		totalUsage.OutputTokens += stepResult.Usage.OutputTokens

		// If no tool calls or no tools configured, we're done.
		if resp.FinishReason != model.FinishReasonToolCalls || len(toolMap) == 0 {
			return buildResult(steps, totalUsage), nil
		}

		// Execute tool calls and append results.
		toolResults := executeToolCalls(ctx, stepResult.ToolCalls, toolMap)
		steps[len(steps)-1].ToolResults = toolResults

		// Append assistant message with tool calls to conversation.
		req.Messages = appendToolMessages(req.Messages, stepResult, toolResults)
	}

	return buildResult(steps, totalUsage), nil
}

// StreamText sends a streaming generation request to the model and
// returns a StreamResult whose Text() method yields text deltas as
// an iterator. The stream must be consumed by ranging over Text() or
// calling Result().
func StreamText(ctx context.Context, m model.Model, opts ...Option) (*StreamResult, error) {
	cfg := resolveConfig(opts)

	var cancel context.CancelFunc
	if cfg.timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, cfg.timeout)
	}

	req := buildRequest(cfg)

	streamResp, err := m.DoStream(ctx, req)
	if err != nil {
		if cancel != nil {
			cancel()
		}

		return nil, err
	}

	return newStreamResult(streamResp.Stream, cancel), nil
}

// resolveConfig applies options to a config with sensible defaults.
func resolveConfig(opts []Option) *config {
	cfg := &config{
		maxSteps: 1,
	}
	for _, opt := range opts {
		opt(cfg)
	}

	return cfg
}

// buildRequest creates a model.Request from resolved config.
func buildRequest(cfg *config) *model.Request {
	var messages []model.Message

	// System message.
	if cfg.system != "" {
		messages = append(messages, model.Message{
			Role:  model.RoleSystem,
			Parts: []model.Part{model.TextPart{Text: cfg.system}},
		})
	}

	// Conversation messages.
	for _, msg := range cfg.messages {
		messages = append(messages, convertMessage(msg))
	}

	// Prompt as a user message (appended after any explicit messages).
	if cfg.prompt != "" {
		messages = append(messages, model.Message{
			Role:  model.RoleUser,
			Parts: []model.Part{model.TextPart{Text: cfg.prompt}},
		})
	}

	req := &model.Request{
		Messages:       messages,
		Temperature:    cfg.temperature,
		MaxTokens:      cfg.maxTokens,
		TopP:           cfg.topP,
		StopSequences:  cfg.stopSequences,
		ResponseFormat: cfg.responseFormat,
	}

	// Attach tool definitions.
	if len(cfg.tools) > 0 {
		for _, t := range cfg.tools {
			req.Tools = append(req.Tools, model.ToolDefinition{
				Name:        t.Name,
				Description: t.Description,
				InputSchema: t.InputSchema,
			})
		}
	}

	return req
}

// convertMessage translates a consumer ai.Message into a model.Message.
func convertMessage(msg Message) model.Message {
	mm := model.Message{
		Role: model.Role(msg.Role),
	}

	// If Parts are set, use them directly.
	if len(msg.Parts) > 0 {
		for _, part := range msg.Parts {
			switch p := part.(type) {
			case TextPart:
				mm.Parts = append(mm.Parts, model.TextPart{Text: p.Text})
			case ImagePart:
				mm.Parts = append(mm.Parts, model.ImagePart{
					URL:      p.URL,
					Data:     p.Data,
					MIMEType: p.MIMEType,
				})
			case ToolCallPart:
				mm.Parts = append(mm.Parts, model.ToolCallPart{
					ToolCallID: p.ToolCallID,
					ToolName:   p.ToolName,
					Args:       p.Args,
				})
			case ToolResultPart:
				mm.Parts = append(mm.Parts, model.ToolResultPart{
					ToolCallID: p.ToolCallID,
					ToolName:   p.ToolName,
					Content:    p.Content,
					IsError:    p.IsError,
				})
			}
		}

		return mm
	}

	// Fall back to Content string as a text part.
	if msg.Content != "" {
		mm.Parts = []model.Part{model.TextPart{Text: msg.Content}}
	}

	return mm
}

// buildToolMap creates a lookup map from tool name to Tool for fast
// execution dispatch.
func buildToolMap(tools []Tool) map[string]Tool {
	if len(tools) == 0 {
		return nil
	}

	m := make(map[string]Tool, len(tools))
	for _, t := range tools {
		m[t.Name] = t
	}

	return m
}

// extractStep converts a model.Response into a Step.
func extractStep(resp *model.Response) Step {
	var text string
	var toolCalls []ToolCall

	for _, c := range resp.Content {
		switch content := c.(type) {
		case model.TextContent:
			text += content.Text
		case model.ToolCallContent:
			toolCalls = append(toolCalls, ToolCall{
				ToolCallID: content.ToolCallID,
				ToolName:   content.ToolName,
				Args:       json.RawMessage(content.Args),
			})
		}
	}

	return Step{
		Text:         text,
		ToolCalls:    toolCalls,
		FinishReason: resp.FinishReason,
		Usage:        resp.Usage,
	}
}

// executeToolCalls runs the tools and returns results.
func executeToolCalls(ctx context.Context, calls []ToolCall, toolMap map[string]Tool) []ToolResult {
	results := make([]ToolResult, 0, len(calls))

	for _, call := range calls {
		tool, ok := toolMap[call.ToolName]
		if !ok {
			results = append(results, ToolResult{
				ToolCallID: call.ToolCallID,
				ToolName:   call.ToolName,
				Content:    fmt.Sprintf("unknown tool: %s", call.ToolName),
				IsError:    true,
			})

			continue
		}

		if tool.Execute == nil {
			results = append(results, ToolResult{
				ToolCallID: call.ToolCallID,
				ToolName:   call.ToolName,
				Content:    fmt.Sprintf("tool %s has no execute function", call.ToolName),
				IsError:    true,
			})

			continue
		}

		content, err := tool.Execute(ctx, call.Args)
		if err != nil {
			results = append(results, ToolResult{
				ToolCallID: call.ToolCallID,
				ToolName:   call.ToolName,
				Content:    err.Error(),
				IsError:    true,
			})

			continue
		}

		results = append(results, ToolResult{
			ToolCallID: call.ToolCallID,
			ToolName:   call.ToolName,
			Content:    content,
		})
	}

	return results
}

// appendToolMessages adds assistant tool calls and tool results to the
// conversation for the next iteration of the tool loop.
func appendToolMessages(msgs []model.Message, step Step, results []ToolResult) []model.Message {
	// Assistant message with tool calls.
	var assistantParts []model.Part
	if step.Text != "" {
		assistantParts = append(assistantParts, model.TextPart{Text: step.Text})
	}
	for _, tc := range step.ToolCalls {
		assistantParts = append(assistantParts, model.ToolCallPart{
			ToolCallID: tc.ToolCallID,
			ToolName:   tc.ToolName,
			Args:       tc.Args,
		})
	}
	msgs = append(msgs, model.Message{
		Role:  model.RoleAssistant,
		Parts: assistantParts,
	})

	// Tool result messages.
	for _, r := range results {
		msgs = append(msgs, model.Message{
			Role: model.RoleTool,
			Parts: []model.Part{
				model.ToolResultPart{
					ToolCallID: r.ToolCallID,
					ToolName:   r.ToolName,
					Content:    r.Content,
					IsError:    r.IsError,
				},
			},
		})
	}

	return msgs
}

// buildResult assembles the final GenerateResult from all steps.
func buildResult(steps []Step, totalUsage Usage) *GenerateResult {
	if len(steps) == 0 {
		return &GenerateResult{}
	}

	last := steps[len(steps)-1]

	// Collect all tool results across all steps.
	var allToolResults []ToolResult
	for _, s := range steps {
		allToolResults = append(allToolResults, s.ToolResults...)
	}

	return &GenerateResult{
		Text:         last.Text,
		ToolCalls:    last.ToolCalls,
		ToolResults:  allToolResults,
		FinishReason: last.FinishReason,
		Usage:        totalUsage,
		Steps:        steps,
	}
}
