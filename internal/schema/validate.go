package schema

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/santhosh-tekuri/jsonschema/v6"
)

const resourceURL = "https://ai-sdk.invalid/schema.json"

// Validator is a compiled JSON Schema. It is safe for concurrent use.
type Validator struct {
	schema *jsonschema.Schema
}

// Compile compiles a schema, defaulting to draft 2020-12. Only references
// within the document resolve; external resources are never loaded. An empty
// schema accepts every value.
func Compile(raw json.RawMessage) (*Validator, error) {
	if len(raw) == 0 {
		raw = json.RawMessage(`{}`)
	}

	document, err := jsonschema.UnmarshalJSON(bytes.NewReader(raw))
	if err != nil {
		return nil, fmt.Errorf("decode schema: %w", err)
	}

	compiler := jsonschema.NewCompiler()
	compiler.DefaultDraft(jsonschema.Draft2020)
	compiler.UseLoader(rejectingLoader{})

	if err := compiler.AddResource(resourceURL, document); err != nil {
		return nil, fmt.Errorf("register schema: %w", err)
	}

	compiled, err := compiler.Compile(resourceURL)
	if err != nil {
		return nil, fmt.Errorf("compile schema: %w", err)
	}

	return &Validator{
		schema: compiled,
	}, nil
}

// Validate checks one JSON value, preserving numeric precision.
func (v *Validator) Validate(raw json.RawMessage) error {
	value, err := jsonschema.UnmarshalJSON(bytes.NewReader(raw))
	if err != nil {
		return fmt.Errorf("decode value: %w", err)
	}

	if err := v.schema.Validate(value); err != nil {
		return fmt.Errorf("validate value: %w", err)
	}

	return nil
}

// rejectingLoader refuses to load external schema resources.
type rejectingLoader struct{}

// Load implements jsonschema.URLLoader.
func (rejectingLoader) Load(url string) (any, error) {
	return nil, fmt.Errorf("external schema %s is not allowed", url)
}
