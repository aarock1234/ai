package ai

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/schema"
	"github.com/aarock1234/ai/pkg/model"
)

// GenerateObject generates a structured response matching type T. It
// auto-generates a JSON Schema from T via reflection, instructs the
// model to produce JSON, and unmarshals the response into T.
//
// This is a top-level generic function because Go does not support
// generic methods on structs.
func GenerateObject[T any](ctx context.Context, m model.Model, opts ...Option) (*ObjectResult[T], error) {
	jsonSchema := schema.Generate[T]()

	if supportsResponseFormat(m, jsonSchema) {
		return generateObjectWithResponseFormat[T](ctx, m, jsonSchema, opts...)
	}

	return generateObjectWithTool[T](ctx, m, jsonSchema, opts...)
}

func generateObjectWithResponseFormat[T any](ctx context.Context, m model.Model, jsonSchema json.RawMessage, opts ...Option) (*ObjectResult[T], error) {
	// Wrap the user's options with JSON schema response format.
	wrappedOpts := make([]Option, len(opts)+1)
	copy(wrappedOpts, opts)
	wrappedOpts[len(opts)] = func(c *config) {
		c.responseFormat = &model.ResponseFormat{
			Type: "json_schema",
			JSONSchema: &model.JSONSchema{
				Name:   "response",
				Schema: jsonSchema,
				Strict: true,
			},
		}
	}

	result, err := GenerateText(ctx, m, wrappedOpts...)
	if err != nil {
		return nil, err
	}

	if result.Text == "" {
		return nil, fmt.Errorf("generate object: %w", ErrNoContent)
	}

	var obj T
	if err := json.Unmarshal([]byte(result.Text), &obj); err != nil {
		return nil, fmt.Errorf("generate object: unmarshalling response: %w", err)
	}

	return &ObjectResult[T]{
		Object:   obj,
		RawJSON:  result.Text,
		Usage:    result.Usage,
		Metadata: result.Metadata,
	}, nil
}

func generateObjectWithTool[T any](ctx context.Context, m model.Model, jsonSchema json.RawMessage, opts ...Option) (*ObjectResult[T], error) {
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
	req.Tools = append(req.Tools, model.ToolDefinition{
		Name:        "structured_output",
		Description: "Return the final response as structured JSON that matches the schema exactly.",
		InputSchema: jsonSchema,
		Strict:      true,
	})
	req.ToolChoice = &model.ToolChoice{
		Type:     "tool",
		ToolName: "structured_output",
	}

	for _, hook := range cfg.prepareStep {
		if err := hook(ctx, 0, req); err != nil {
			return nil, fmt.Errorf("prepare step 0: %w", err)
		}
	}

	resp, err := doGenerate(ctx, m, req, cfg.maxRetries)
	if err != nil {
		return nil, fmt.Errorf("generate object: %w", err)
	}

	step := extractStep(resp)
	for _, hook := range cfg.onStepFinish {
		if err := hook(ctx, step); err != nil {
			return nil, fmt.Errorf("on step finish 0: %w", err)
		}
	}

	if _, err := finalizeResult(ctx, cfg.onFinish, buildResult([]Step{step}, step.Usage)); err != nil {
		return nil, err
	}

	raw := step.Text
	if len(step.ToolCalls) > 0 {
		raw = string(step.ToolCalls[0].Args)
	}

	if raw == "" {
		return nil, fmt.Errorf("generate object: %w", ErrNoContent)
	}

	var obj T
	if err := json.Unmarshal([]byte(raw), &obj); err != nil {
		return nil, fmt.Errorf("generate object: unmarshalling response: %w", err)
	}

	return &ObjectResult[T]{
		Object:   obj,
		RawJSON:  raw,
		Usage:    step.Usage,
		Metadata: step.Metadata,
	}, nil
}

func supportsResponseFormat(m model.Model, jsonSchema json.RawMessage) bool {
	support, ok := m.(model.ResponseFormatSupport)
	if !ok {
		return false
	}

	return support.SupportsResponseFormat(&model.ResponseFormat{
		Type: "json_schema",
		JSONSchema: &model.JSONSchema{
			Name:   "response",
			Schema: jsonSchema,
			Strict: true,
		},
	})
}
