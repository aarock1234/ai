package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/aarock1234/ai/internal/partialjson"
	"github.com/aarock1234/ai/internal/schema"
	"github.com/aarock1234/ai/pkg/model"
)

// draft07SchemaURL identifies the JSON Schema dialect of generated wrapper
// schemas.
const draft07SchemaURL = "http://json-schema.org/draft-07/schema#"

// Output specifies the shape of the generated output: plain text, a typed
// object, an array, a choice or arbitrary JSON. Use TextOutput, ObjectOutput,
// ArrayOutput, ChoiceOutput or JSONOutput, or implement it for a custom
// format.
type Output interface {
	// Name identifies the output mode, such as "text" or "object".
	Name() string

	// ResponseFormat returns the response format requested from the model.
	// An error means the output was configured incorrectly; generation
	// reports it before calling the model.
	ResponseFormat() (*model.ResponseFormat, error)

	// ParseCompleteOutput parses the text of the final step.
	ParseCompleteOutput(ctx context.Context, text string, outputContext OutputContext) (any, error)

	// ParsePartialOutput parses streamed text that may be incomplete. It
	// reports false when no partial value can be derived yet.
	ParsePartialOutput(text string) (any, bool)
}

// OutputContext describes the step whose text is parsed, for error reports.
type OutputContext struct {
	Response     model.ResponseInfo
	Usage        LanguageModelUsage
	FinishReason model.UnifiedFinishReason
}

// ElementOutput is implemented by outputs whose partial outputs contain
// elements that can be streamed as they complete, such as ArrayOutput. It
// is the Go form of upstream's createElementStreamTransform.
type ElementOutput interface {
	Output

	// ElementPublisher returns a function that receives each partial output
	// of one stream and returns the elements completed since the previous
	// call, or an error when the elements violate the output's constraints.
	ElementPublisher() func(partial any) ([]any, error)
}

// ObjectOutputOptions configures ObjectOutput.
type ObjectOutputOptions struct {
	// Name and Description optionally guide the model, for example as the
	// schema name.
	Name        string
	Description string

	// Schema overrides the JSON Schema generated from the Go type.
	Schema json.RawMessage
}

// ArrayOutputOptions configures ArrayOutput.
type ArrayOutputOptions struct {
	// MinItems and MaxItems optionally bound the number of elements.
	MinItems *int
	MaxItems *int

	Name        string
	Description string

	// ElementSchema overrides the JSON Schema generated from the element
	// type.
	ElementSchema json.RawMessage
}

// ChoiceOutputOptions configures ChoiceOutput.
type ChoiceOutputOptions struct {
	// Options are the values the model chooses from.
	Options []string

	Name        string
	Description string
}

// JSONOutputOptions configures JSONOutput.
type JSONOutputOptions struct {
	Name        string
	Description string
}

// TextOutput returns the default output mode: plain text. Its output is a
// string.
func TextOutput() Output {
	return textOutput{}
}

// ObjectOutput returns an output mode that generates a JSON object matching
// the schema of T and decodes it into a T. Partial outputs are T values
// decoded from the incomplete JSON without validation.
func ObjectOutput[T any](options ObjectOutputOptions) Output {
	output := &objectOutput[T]{
		name:        options.Name,
		description: options.Description,
	}

	output.schema, output.validator, output.err = compileOutputSchema[T](options.Schema)

	return output
}

// ArrayOutput returns an output mode that generates an array of elements
// matching the schema of T. Its output and partial outputs are []T values;
// partial outputs contain only complete, valid elements. Streams can publish
// each element as it completes.
func ArrayOutput[T any](options ArrayOutputOptions) Output {
	output := &arrayOutput[T]{
		minItems:    options.MinItems,
		maxItems:    options.MaxItems,
		name:        options.Name,
		description: options.Description,
	}

	if err := validateArrayBounds(options.MinItems, options.MaxItems); err != nil {
		output.err = err

		return output
	}

	output.elementSchema, output.validator, output.err = compileOutputSchema[T](options.ElementSchema)

	return output
}

// ChoiceOutput returns an output mode that selects one of the options. Its
// output is the chosen string.
func ChoiceOutput(options ChoiceOutputOptions) Output {
	return &choiceOutput{
		options:     slices.Clone(options.Options),
		name:        options.Name,
		description: options.Description,
	}
}

// JSONOutput returns an output mode that generates any JSON value. Its output
// is a json.RawMessage.
func JSONOutput(options JSONOutputOptions) Output {
	return &jsonOutput{
		name:        options.Name,
		description: options.Description,
	}
}

// OutputAs returns a parsed output as a T. It reports a
// *NoOutputGeneratedError when output is nil, which means that no output was
// parsed.
func OutputAs[T any](output any) (T, error) {
	var zero T

	if output == nil {
		return zero, &NoOutputGeneratedError{}
	}

	typed, ok := output.(T)
	if !ok {
		return zero, fmt.Errorf("output has type %T, not %T", output, zero)
	}

	return typed, nil
}

type textOutput struct{}

func (textOutput) Name() string {
	return "text"
}

func (textOutput) ResponseFormat() (*model.ResponseFormat, error) {
	return &model.ResponseFormat{
		Type: model.ResponseFormatText,
	}, nil
}

func (textOutput) ParseCompleteOutput(_ context.Context, text string, _ OutputContext) (any, error) {
	return text, nil
}

func (textOutput) ParsePartialOutput(text string) (any, bool) {
	return text, true
}

type objectOutput[T any] struct {
	name        string
	description string
	schema      json.RawMessage
	validator   *schema.Validator
	err         error
}

func (o *objectOutput[T]) Name() string {
	return "object"
}

func (o *objectOutput[T]) ResponseFormat() (*model.ResponseFormat, error) {
	if o.err != nil {
		return nil, o.err
	}

	return &model.ResponseFormat{
		Type:        model.ResponseFormatJSON,
		Schema:      o.schema,
		Name:        o.name,
		Description: o.description,
	}, nil
}

func (o *objectOutput[T]) ParseCompleteOutput(_ context.Context, text string, outputContext OutputContext) (any, error) {
	raw, err := parseOutputJSON(text)
	if err != nil {
		return nil, unparsableOutputError(text, outputContext, err)
	}

	if err := o.validator.Validate(raw); err != nil {
		return nil, mismatchedOutputError(text, outputContext, err)
	}

	var value T
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil, mismatchedOutputError(text, outputContext, err)
	}

	return value, nil
}

func (o *objectOutput[T]) ParsePartialOutput(text string) (any, bool) {
	raw, ok := parsePartialOutputJSON(text)
	if !ok {
		return nil, false
	}

	var value T
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil, false
	}

	return value, true
}

type arrayOutput[T any] struct {
	minItems      *int
	maxItems      *int
	name          string
	description   string
	elementSchema json.RawMessage
	validator     *schema.Validator
	err           error
}

func (o *arrayOutput[T]) Name() string {
	return "array"
}

func (o *arrayOutput[T]) ResponseFormat() (*model.ResponseFormat, error) {
	if o.err != nil {
		return nil, o.err
	}

	wrapper, err := wrapArraySchema(arrayWrapperSchemaOptions{
		elementSchema: o.elementSchema,
		minItems:      o.minItems,
		maxItems:      o.maxItems,
	})
	if err != nil {
		return nil, err
	}

	return &model.ResponseFormat{
		Type:        model.ResponseFormatJSON,
		Schema:      wrapper,
		Name:        o.name,
		Description: o.description,
	}, nil
}

func (o *arrayOutput[T]) ParseCompleteOutput(_ context.Context, text string, outputContext OutputContext) (any, error) {
	raw, err := parseOutputJSON(text)
	if err != nil {
		return nil, unparsableOutputError(text, outputContext, err)
	}

	rawElements, ok := objectArrayProperty(raw, "elements")
	if !ok {
		return nil, mismatchedOutputError(text, outputContext, errors.New("response must be an object with an elements array"))
	}

	if err := arrayLengthError(len(rawElements), o.minItems, o.maxItems); err != nil {
		return nil, mismatchedOutputError(text, outputContext, err)
	}

	elements := make([]T, 0, len(rawElements))
	for _, rawElement := range rawElements {
		element, err := o.decodeElement(rawElement)
		if err != nil {
			return nil, mismatchedOutputError(text, outputContext, err)
		}

		elements = append(elements, element)
	}

	return elements, nil
}

// ParsePartialOutput returns the complete, valid elements. The last element
// of repaired JSON may still be incomplete and is left out.
func (o *arrayOutput[T]) ParsePartialOutput(text string) (any, bool) {
	raw, state := partialjson.Parse(text)
	if raw == nil {
		return nil, false
	}

	rawElements, ok := objectArrayProperty(raw, "elements")
	if !ok {
		return nil, false
	}

	if state == partialjson.Repaired && len(rawElements) > 0 {
		rawElements = rawElements[:len(rawElements)-1]
	}

	elements := make([]T, 0, len(rawElements))
	for _, rawElement := range rawElements {
		element, err := o.decodeElement(rawElement)
		if err == nil {
			elements = append(elements, element)
		}
	}

	return elements, true
}

func (o *arrayOutput[T]) decodeElement(raw json.RawMessage) (T, error) {
	var element T

	if err := o.validator.Validate(raw); err != nil {
		return element, err
	}

	if err := json.Unmarshal(raw, &element); err != nil {
		return element, fmt.Errorf("decode element: %w", err)
	}

	return element, nil
}

// ElementPublisher implements ElementOutput.
func (o *arrayOutput[T]) ElementPublisher() func(partial any) ([]any, error) {
	published := 0

	return func(partial any) ([]any, error) {
		elements, ok := partial.([]T)
		if !ok {
			return nil, nil
		}

		var completed []any
		for ; published < len(elements); published++ {
			if o.maxItems != nil && published >= *o.maxItems {
				return completed, arrayLengthError(len(elements), nil, o.maxItems)
			}

			completed = append(completed, elements[published])
		}

		return completed, nil
	}
}

type choiceOutput struct {
	options     []string
	name        string
	description string
}

func (o *choiceOutput) Name() string {
	return "choice"
}

// choiceSchema is the object schema that wraps the choice in a "result"
// property. Its fields follow sorted key order.
type choiceSchema struct {
	Schema               string           `json:"$schema"`
	AdditionalProperties bool             `json:"additionalProperties"`
	Properties           choiceProperties `json:"properties"`
	Required             []string         `json:"required"`
	Type                 string           `json:"type"`
}

type choiceProperties struct {
	Result choiceResultSchema `json:"result"`
}

type choiceResultSchema struct {
	Enum []string `json:"enum"`
	Type string   `json:"type"`
}

func (o *choiceOutput) ResponseFormat() (*model.ResponseFormat, error) {
	wrapper, err := json.Marshal(choiceSchema{
		Schema:               draft07SchemaURL,
		AdditionalProperties: false,
		Properties: choiceProperties{
			Result: choiceResultSchema{
				Enum: o.options,
				Type: "string",
			},
		},
		Required: []string{
			"result",
		},
		Type: "object",
	})
	if err != nil {
		return nil, fmt.Errorf("encode choice schema: %w", err)
	}

	return &model.ResponseFormat{
		Type:        model.ResponseFormatJSON,
		Schema:      wrapper,
		Name:        o.name,
		Description: o.description,
	}, nil
}

func (o *choiceOutput) ParseCompleteOutput(_ context.Context, text string, outputContext OutputContext) (any, error) {
	raw, err := parseOutputJSON(text)
	if err != nil {
		return nil, unparsableOutputError(text, outputContext, err)
	}

	choice, ok := objectStringProperty(raw, "result")
	if !ok || !slices.Contains(o.options, choice) {
		return nil, mismatchedOutputError(text, outputContext, errors.New("response must be an object that contains a choice value"))
	}

	return choice, nil
}

// ParsePartialOutput returns an exact choice for complete JSON and, for
// repaired JSON, the only option that starts with the partial value.
func (o *choiceOutput) ParsePartialOutput(text string) (any, bool) {
	raw, state := partialjson.Parse(text)
	if raw == nil {
		return nil, false
	}

	partial, ok := objectStringProperty(raw, "result")
	if !ok {
		return nil, false
	}

	matches := slices.DeleteFunc(slices.Clone(o.options), func(option string) bool {
		return !strings.HasPrefix(option, partial)
	})

	if state == partialjson.Successful {
		return partial, slices.Contains(matches, partial)
	}

	if len(matches) != 1 {
		return nil, false
	}

	return matches[0], true
}

type jsonOutput struct {
	name        string
	description string
}

func (o *jsonOutput) Name() string {
	return "json"
}

func (o *jsonOutput) ResponseFormat() (*model.ResponseFormat, error) {
	return &model.ResponseFormat{
		Type:        model.ResponseFormatJSON,
		Name:        o.name,
		Description: o.description,
	}, nil
}

func (o *jsonOutput) ParseCompleteOutput(_ context.Context, text string, outputContext OutputContext) (any, error) {
	raw, err := parseOutputJSON(text)
	if err != nil {
		return nil, unparsableOutputError(text, outputContext, err)
	}

	return raw, nil
}

func (o *jsonOutput) ParsePartialOutput(text string) (any, bool) {
	raw, ok := parsePartialOutputJSON(text)
	if !ok {
		return nil, false
	}

	return raw, true
}

// compileOutputSchema returns the override schema, or the schema generated
// from T, with its validator.
func compileOutputSchema[T any](override json.RawMessage) (json.RawMessage, *schema.Validator, error) {
	outputSchema := override
	if len(outputSchema) == 0 {
		generated, err := schema.Generate[T]()
		if err != nil {
			return nil, nil, fmt.Errorf("generate output schema: %w", err)
		}

		outputSchema = generated
	}

	validator, err := schema.Compile(outputSchema)
	if err != nil {
		return nil, nil, fmt.Errorf("compile output schema: %w", err)
	}

	return outputSchema, validator, nil
}

// arrayWrapperSchemaOptions describe the object schema that wraps array
// elements.
type arrayWrapperSchemaOptions struct {
	elementSchema json.RawMessage
	minItems      *int
	maxItems      *int
}

// arrayWrapperSchema is the object schema that wraps array elements. Its
// fields follow sorted key order.
type arrayWrapperSchema struct {
	Defs                 json.RawMessage        `json:"$defs,omitempty"`
	Schema               string                 `json:"$schema"`
	AdditionalProperties bool                   `json:"additionalProperties"`
	Definitions          json.RawMessage        `json:"definitions,omitempty"`
	Properties           arrayWrapperProperties `json:"properties"`
	Required             []string               `json:"required"`
	Type                 string                 `json:"type"`
}

type arrayWrapperProperties struct {
	Elements arrayElementsSchema `json:"elements"`
}

type arrayElementsSchema struct {
	Items    map[string]json.RawMessage `json:"items"`
	MaxItems *int                       `json:"maxItems,omitempty"`
	MinItems *int                       `json:"minItems,omitempty"`
	Type     string                     `json:"type"`
}

// wrapArraySchema wraps the element schema in an object with an elements
// array, because providers require object schemas at the root. Root-level
// definitions move to the wrapper so that root-relative references resolve.
func wrapArraySchema(options arrayWrapperSchemaOptions) (json.RawMessage, error) {
	var itemSchema map[string]json.RawMessage
	if err := json.Unmarshal(options.elementSchema, &itemSchema); err != nil {
		return nil, fmt.Errorf("decode element schema: %w", err)
	}

	wrapper := arrayWrapperSchema{
		Defs:                 takeSchemaKey(itemSchema, "$defs"),
		Schema:               draft07SchemaURL,
		AdditionalProperties: false,
		Definitions:          takeSchemaKey(itemSchema, "definitions"),
		Properties: arrayWrapperProperties{
			Elements: arrayElementsSchema{
				Items:    itemSchema,
				MaxItems: options.maxItems,
				MinItems: options.minItems,
				Type:     "array",
			},
		},
		Required: []string{
			"elements",
		},
		Type: "object",
	}

	delete(itemSchema, "$schema")

	encoded, err := json.Marshal(wrapper)
	if err != nil {
		return nil, fmt.Errorf("encode array schema: %w", err)
	}

	return encoded, nil
}

// takeSchemaKey removes key from a schema object and returns its value.
func takeSchemaKey(schemaObject map[string]json.RawMessage, key string) json.RawMessage {
	value, ok := schemaObject[key]
	if !ok {
		return nil
	}

	delete(schemaObject, key)

	return value
}

func validateArrayBounds(minItems, maxItems *int) error {
	if minItems != nil && *minItems < 0 {
		return &InvalidArgumentError{
			Parameter: "minItems",
			Value:     *minItems,
			Message:   "minItems must be greater than or equal to 0",
		}
	}

	if maxItems != nil && *maxItems < 0 {
		return &InvalidArgumentError{
			Parameter: "maxItems",
			Value:     *maxItems,
			Message:   "maxItems must be greater than or equal to 0",
		}
	}

	if minItems != nil && maxItems != nil && *minItems > *maxItems {
		return &InvalidArgumentError{
			Parameter: "minItems",
			Value:     *minItems,
			Message:   "minItems must be less than or equal to maxItems",
		}
	}

	return nil
}

func arrayLengthError(length int, minItems, maxItems *int) error {
	if minItems != nil && length < *minItems {
		return fmt.Errorf("elements array must contain at least %d items", *minItems)
	}

	if maxItems != nil && length > *maxItems {
		return fmt.Errorf("elements array must contain at most %d items", *maxItems)
	}

	return nil
}

// errInvalidOutputJSON reports output text that is not one JSON value.
var errInvalidOutputJSON = errors.New("output is not valid JSON")

// parseOutputJSON checks that text is one JSON value.
func parseOutputJSON(text string) (json.RawMessage, error) {
	raw := json.RawMessage(text)
	if !json.Valid(raw) {
		return nil, errInvalidOutputJSON
	}

	return raw, nil
}

// parsePartialOutputJSON parses complete or repairable JSON text.
func parsePartialOutputJSON(text string) (json.RawMessage, bool) {
	raw, _ := partialjson.Parse(text)

	return raw, raw != nil
}

// objectArrayProperty returns the array stored under key in a JSON object.
func objectArrayProperty(raw json.RawMessage, key string) ([]json.RawMessage, bool) {
	var object map[string]json.RawMessage
	if err := json.Unmarshal(raw, &object); err != nil || object == nil {
		return nil, false
	}

	property := bytes.TrimSpace(object[key])
	if len(property) == 0 || property[0] != '[' {
		return nil, false
	}

	var elements []json.RawMessage
	if err := json.Unmarshal(property, &elements); err != nil {
		return nil, false
	}

	return elements, true
}

// objectStringProperty returns the string stored under key in a JSON object.
func objectStringProperty(raw json.RawMessage, key string) (string, bool) {
	var object map[string]json.RawMessage
	if err := json.Unmarshal(raw, &object); err != nil || object == nil {
		return "", false
	}

	property := bytes.TrimSpace(object[key])
	if len(property) == 0 || property[0] != '"' {
		return "", false
	}

	var value string
	if err := json.Unmarshal(property, &value); err != nil {
		return "", false
	}

	return value, true
}

func unparsableOutputError(text string, outputContext OutputContext, cause error) error {
	return &NoObjectGeneratedError{
		Message:      "no object generated: could not parse the response",
		Text:         text,
		Response:     outputContext.Response,
		Usage:        outputContext.Usage,
		FinishReason: outputContext.FinishReason,
		Cause:        cause,
	}
}

func mismatchedOutputError(text string, outputContext OutputContext, cause error) error {
	return &NoObjectGeneratedError{
		Message:      "no object generated: response did not match schema",
		Text:         text,
		Response:     outputContext.Response,
		Usage:        outputContext.Usage,
		FinishReason: outputContext.FinishReason,
		Cause:        cause,
	}
}
