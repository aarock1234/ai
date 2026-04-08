package ai

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/schema"
)

// GenerateObject generates a structured response matching type T. It
// auto-generates a JSON Schema from T via reflection, instructs the
// model to produce JSON, and unmarshals the response into T.
//
// This is a top-level generic function because Go does not support
// generic methods on structs.
func GenerateObject[T any](ctx context.Context, m model.Model, opts ...Option) (*ObjectResult[T], error) {
	jsonSchema := schema.Generate[T]()

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
		Object: obj,
		Usage:  result.Usage,
	}, nil
}
