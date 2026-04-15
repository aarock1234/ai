// Package ai provides the high-level functions for interacting with
// language models through the AI SDK. Consumers import this package
// alongside a provider package (e.g. pkg/provider/openai) to generate
// text, stream responses, produce structured output, and embed text.
package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apierror"
	"github.com/aarock1234/ai/pkg/model"
)

// GenerateText sends a non-streaming generation request to the model
// and returns the complete result. When tools are provided and
// WithMaxSteps is greater than 1, GenerateText runs the automatic
// tool loop: it calls tools, feeds results back to the model, and
// repeats until the model stops or the step limit is reached.
func GenerateText(ctx context.Context, m model.Model, opts ...Option) (*GenerateResult, error) {
	cfg, err := resolveConfig(opts)
	if err != nil {
		return nil, err
	}

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
		for _, hook := range cfg.prepareStep {
			if err := hook(ctx, step, req); err != nil {
				return nil, fmt.Errorf("prepare step %d: %w", step, err)
			}
		}

		resp, err := doGenerate(ctx, m, req, cfg.maxRetries)
		if err != nil {
			return nil, fmt.Errorf("step %d: %w", step, err)
		}

		stepResult := extractStep(resp)
		steps = append(steps, stepResult)
		addUsage(&totalUsage, stepResult.Usage)

		for _, hook := range cfg.onStepFinish {
			if err := hook(ctx, stepResult); err != nil {
				return nil, fmt.Errorf("on step finish %d: %w", step, err)
			}
		}

		if shouldStop(cfg.stopWhen, stepResult) {
			return finalizeResult(ctx, cfg.onFinish, buildResult(steps, totalUsage))
		}

		// If no tool calls or no tools configured, we're done.
		if resp.FinishReason != model.FinishReasonToolCalls || len(toolMap) == 0 {
			return finalizeResult(ctx, cfg.onFinish, buildResult(steps, totalUsage))
		}

		// Execute tool calls and append results.
		toolResults := executeToolCalls(ctx, stepResult.ToolCalls, toolMap)
		steps[len(steps)-1].ToolResults = toolResults

		// Append assistant message with tool calls to conversation.
		req.Messages = appendToolMessages(req.Messages, stepResult, toolResults)
	}

	return finalizeResult(ctx, cfg.onFinish, buildResult(steps, totalUsage))
}

// StreamText sends a streaming generation request to the model and
// returns a StreamResult whose Text() method yields text deltas as
// an iterator. The stream must be consumed by ranging over Text() or
// calling Result().
//
// When tools are provided and WithMaxSteps is greater than 1,
// StreamText runs the same automatic tool loop as GenerateText:
// after each streamed step that finishes with tool calls, it executes
// the tools, feeds results back to the model, and streams the next
// step — all transparently through a single StreamResult.
func StreamText(ctx context.Context, m model.Model, opts ...Option) (*StreamResult, error) {
	cfg, err := resolveConfig(opts)
	if err != nil {
		return nil, err
	}

	var cancel context.CancelFunc
	if cfg.timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, cfg.timeout)
	}

	req := buildRequest(cfg)
	toolMap := buildToolMap(cfg.tools)
	finishHooks := bindFinishHooks(ctx, cfg.onFinish)

	// Multi-step tool loop: build a stream that chains provider
	// streams across steps, executing tools between them.
	if len(toolMap) > 0 && cfg.maxSteps > 1 {
		state := &toolLoopState{}
		stream := streamToolLoop(ctx, m, req, cfg, toolMap, state)

		return newStreamResult(stream, model.ProviderMetadata{}, cancel, finishHooks, state), nil
	}

	// Single-step: stream one provider response directly.
	for _, hook := range cfg.prepareStep {
		if err := hook(ctx, 0, req); err != nil {
			if cancel != nil {
				cancel()
			}

			return nil, fmt.Errorf("prepare stream: %w", err)
		}
	}

	streamResp, err := doStream(ctx, m, req, cfg.maxRetries)
	if err != nil {
		if cancel != nil {
			cancel()
		}

		return nil, err
	}

	return newStreamResult(streamResp.Stream, streamResp.Metadata, cancel, finishHooks, nil), nil
}

// resolveConfig applies options to a config with sensible defaults.
func resolveConfig(opts []Option) (*config, error) {
	cfg := &config{
		maxSteps: 1,
	}
	for _, opt := range opts {
		opt(cfg)
	}

	if cfg.maxSteps < 1 {
		return nil, ErrInvalidMaxSteps
	}
	if cfg.err != nil {
		return nil, cfg.err
	}

	return cfg, nil
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
		Messages:         messages,
		Temperature:      cfg.temperature,
		MaxTokens:        cfg.maxTokens,
		TopP:             cfg.topP,
		TopK:             cfg.topK,
		PresencePenalty:  cfg.presencePenalty,
		FrequencyPenalty: cfg.frequencyPenalty,
		Seed:             cfg.seed,
		StopSequences:    append([]string(nil), cfg.stopSequences...),
		ToolChoice:       cloneToolChoice(cfg.toolChoice),
		ResponseFormat:   cloneResponseFormat(cfg.responseFormat),
		Headers:          cloneHeaders(cfg.headers),
		ProviderOptions:  cfg.providerOpts.Clone(),
	}

	// Attach tool definitions.
	if len(cfg.tools) > 0 {
		activeTools := buildActiveToolSet(cfg.activeTools)
		for _, t := range cfg.tools {
			if len(activeTools) > 0 && !activeTools[t.Name] {
				continue
			}
			req.Tools = append(req.Tools, model.ToolDefinition{
				Name:        t.Name,
				Description: t.Description,
				InputSchema: t.InputSchema,
				Strict:      t.Strict,
			})
		}
		req.ActiveTools = append([]string(nil), cfg.activeTools...)
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
		Text:            text,
		Reasoning:       resp.Reasoning,
		ToolCalls:       toolCalls,
		FinishReason:    resp.FinishReason,
		RawFinishReason: resp.RawFinishReason,
		Usage:           resp.Usage,
		Metadata:        resp.Metadata.Clone(),
		Warnings:        append([]string(nil), resp.Warnings...),
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
		Text:            last.Text,
		Reasoning:       last.Reasoning,
		ToolCalls:       last.ToolCalls,
		ToolResults:     allToolResults,
		FinishReason:    last.FinishReason,
		RawFinishReason: last.RawFinishReason,
		Usage:           totalUsage,
		Steps:           steps,
		Metadata:        last.Metadata.Clone(),
		Warnings:        append([]string(nil), last.Warnings...),
	}
}

func buildActiveToolSet(names []string) map[string]bool {
	if len(names) == 0 {
		return nil
	}

	active := make(map[string]bool, len(names))
	for _, name := range names {
		active[name] = true
	}

	return active
}

func cloneToolChoice(choice *model.ToolChoice) *model.ToolChoice {
	if choice == nil {
		return nil
	}

	return &model.ToolChoice{
		Type:     choice.Type,
		ToolName: choice.ToolName,
	}
}

func cloneResponseFormat(rf *model.ResponseFormat) *model.ResponseFormat {
	if rf == nil {
		return nil
	}

	cloned := &model.ResponseFormat{
		Type: rf.Type,
	}

	if rf.JSONSchema != nil {
		cloned.JSONSchema = &model.JSONSchema{
			Name:   rf.JSONSchema.Name,
			Schema: append(json.RawMessage(nil), rf.JSONSchema.Schema...),
			Strict: rf.JSONSchema.Strict,
		}
	}

	return cloned
}

func cloneHeaders(headers http.Header) http.Header {
	if len(headers) == 0 {
		return nil
	}

	return headers.Clone()
}

func shouldStop(conditions []StopWhenFunc, step Step) bool {
	for _, fn := range conditions {
		if fn(step) {
			return true
		}
	}

	return false
}

func finalizeResult(ctx context.Context, hooks []OnFinishFunc, result *GenerateResult) (*GenerateResult, error) {
	for _, hook := range hooks {
		if err := hook(ctx, result); err != nil {
			return nil, fmt.Errorf("on finish: %w", err)
		}
	}

	return result, nil
}

type finishHook func(*GenerateResult) error

type stepHook func(Step) error

func bindFinishHooks(ctx context.Context, hooks []OnFinishFunc) []finishHook {
	if len(hooks) == 0 {
		return nil
	}

	bound := make([]finishHook, len(hooks))
	for i, hook := range hooks {
		bound[i] = func(result *GenerateResult) error {
			return hook(ctx, result)
		}
	}

	return bound
}

func bindStepHooks(ctx context.Context, hooks []OnStepFinishFunc) []stepHook {
	if len(hooks) == 0 {
		return nil
	}

	bound := make([]stepHook, len(hooks))
	for i, hook := range hooks {
		bound[i] = func(step Step) error {
			return hook(ctx, step)
		}
	}

	return bound
}

func runStepHooks(hooks []stepHook, step Step) error {
	for _, hook := range hooks {
		if err := hook(step); err != nil {
			return err
		}
	}

	return nil
}

func runFinishHooks(hooks []finishHook, result *GenerateResult) error {
	for _, hook := range hooks {
		if err := hook(result); err != nil {
			return err
		}
	}

	return nil
}

func doGenerate(ctx context.Context, m model.Model, req *model.Request, maxRetries int) (*model.Response, error) {
	var lastErr error

	for attempt := range maxRetries + 1 {
		resp, err := m.DoGenerate(ctx, req.Clone())
		if err == nil {
			return resp, nil
		}

		lastErr = err
		if attempt == maxRetries || !isRetryableError(err) {
			return nil, err
		}

		if err := waitForRetry(ctx, attempt); err != nil {
			return nil, err
		}
	}

	return nil, lastErr
}

func doStream(ctx context.Context, m model.Model, req *model.Request, maxRetries int) (*model.StreamResponse, error) {
	var lastErr error

	for attempt := range maxRetries + 1 {
		resp, err := m.DoStream(ctx, req.Clone())
		if err == nil {
			return resp, nil
		}

		lastErr = err
		if attempt == maxRetries || !isRetryableError(err) {
			return nil, err
		}

		if err := waitForRetry(ctx, attempt); err != nil {
			return nil, err
		}
	}

	return nil, lastErr
}

// streamToolLoop returns a stream iterator that transparently chains
// multiple provider streams across tool-loop steps. For each step it
// starts a new stream, yields all chunks to the caller, and — when
// the step finishes with tool calls — executes the tools, appends
// the results to the conversation, and continues with the next step.
//
// The shared toolLoopState is populated with completed steps so that
// StreamResult can build the correct multi-step GenerateResult.
func streamToolLoop(ctx context.Context, m model.Model, req *model.Request, cfg *config, toolMap map[string]Tool, state *toolLoopState) iter.Seq[model.Chunk] {
	return func(yield func(model.Chunk) bool) {
		for step := range cfg.maxSteps {
			for _, hook := range cfg.prepareStep {
				if err := hook(ctx, step, req); err != nil {
					state.err = fmt.Errorf("prepare step %d: %w", step, err)

					return
				}
			}

			streamResp, err := doStream(ctx, m, req, cfg.maxRetries)
			if err != nil {
				state.err = fmt.Errorf("step %d: %w", step, err)

				return
			}

			// Accumulate this step's data from the stream while
			// forwarding every chunk to the caller.
			var (
				textBuilder      strings.Builder
				reasoningBuilder strings.Builder
				finishReason     model.FinishReason
				rawFinishReason  string
				stepUsage        Usage
				stepMetadata     = streamResp.Metadata.Clone()
				warnings         []string
			)

			toolCallArgs := make(map[string]*strings.Builder)
			toolCallNames := make(map[string]string)

			for chunk := range streamResp.Stream {
				switch c := chunk.(type) {
				case model.TextDelta:
					textBuilder.WriteString(c.Text)
				case model.ReasoningDelta:
					reasoningBuilder.WriteString(c.Text)
				case model.ToolCallDelta:
					if _, ok := toolCallArgs[c.ToolCallID]; !ok {
						toolCallArgs[c.ToolCallID] = &strings.Builder{}
					}
					toolCallArgs[c.ToolCallID].WriteString(c.ArgsJSON)
					if c.ToolName != "" {
						toolCallNames[c.ToolCallID] = c.ToolName
					}
				case model.FinishChunk:
					finishReason = c.FinishReason
					rawFinishReason = c.RawFinishReason
					if rawFinishReason == "" {
						rawFinishReason = string(c.FinishReason)
					}
				case model.UsageChunk:
					stepUsage = c.Usage
				case model.ProviderMetadataChunk:
					stepMetadata = c.Metadata.Clone()
				}

				if !yield(chunk) {
					return
				}
			}

			// Assemble the completed step.
			var toolCalls []ToolCall
			for id, args := range toolCallArgs {
				toolCalls = append(toolCalls, ToolCall{
					ToolCallID: id,
					ToolName:   toolCallNames[id],
					Args:       []byte(args.String()),
				})
			}

			stepResult := Step{
				Text:            textBuilder.String(),
				Reasoning:       reasoningBuilder.String(),
				ToolCalls:       toolCalls,
				FinishReason:    finishReason,
				RawFinishReason: rawFinishReason,
				Usage:           stepUsage,
				Metadata:        stepMetadata,
				Warnings:        warnings,
			}

			// Fire step hooks.
			for _, hook := range cfg.onStepFinish {
				if err := hook(ctx, stepResult); err != nil {
					state.err = fmt.Errorf("on step finish %d: %w", step, err)
					state.recordStep(stepResult)

					return
				}
			}

			// Check custom stop conditions.
			if shouldStop(cfg.stopWhen, stepResult) {
				state.recordStep(stepResult)

				return
			}

			// No tool calls or no tools → final step.
			if finishReason != model.FinishReasonToolCalls || len(toolMap) == 0 {
				state.recordStep(stepResult)

				return
			}

			// Execute tools and record the step with results.
			toolResults := executeToolCalls(ctx, stepResult.ToolCalls, toolMap)
			stepResult.ToolResults = toolResults
			state.recordStep(stepResult)

			// Append assistant + tool messages for the next iteration.
			req.Messages = appendToolMessages(req.Messages, stepResult, toolResults)
		}
	}
}

func isRetryableError(err error) bool {
	var apiErr *apierror.APIError
	if errors.As(err, &apiErr) {
		return apiErr.IsRetryable
	}

	return false
}

func waitForRetry(ctx context.Context, attempt int) error {
	backoff := 100 * time.Millisecond
	for range attempt {
		backoff *= 2
	}

	timer := time.NewTimer(backoff)
	defer timer.Stop()

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
