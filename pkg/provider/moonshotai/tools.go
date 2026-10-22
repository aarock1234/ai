package moonshotai

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// requiredToolChoiceRejected lists the models that reject tool choice
// "required".
var requiredToolChoiceRejected = map[string]bool{
	"kimi-k2.6":                true,
	"kimi-k2.7-code":           true,
	"kimi-k2.7-code-highspeed": true,
}

// preparedTools are the tools and tool choice of a request.
type preparedTools struct {
	tools      []functionTool
	toolChoice *toolChoice
	warnings   []model.Warning
}

// prepareTools converts function tools and the tool choice. Provider tools
// are not supported.
func prepareTools(tools []model.Tool, choice *model.ToolChoice, modelID string) (preparedTools, error) {
	if len(tools) == 0 {
		return preparedTools{}, nil
	}

	var prepared preparedTools
	prepared.tools = []functionTool{}

	for _, tool := range tools {
		switch tool := tool.(type) {
		case model.FunctionTool:
			converted, err := convertFunctionTool(functionToolSpec{
				name:        tool.Name,
				description: tool.Description,
				inputSchema: tool.InputSchema,
				strict:      tool.Strict,
			})
			if err != nil {
				return preparedTools{}, err
			}

			prepared.tools = append(prepared.tools, converted)

		case model.ProviderTool:
			prepared.warnings = append(prepared.warnings, model.UnsupportedWarning("provider-defined tool "+tool.ID, ""))
		}
	}

	if choice == nil {
		return prepared, nil
	}

	switch choice.Type {
	case model.ToolChoiceAuto, model.ToolChoiceNone:
		prepared.toolChoice = &toolChoice{
			mode: string(choice.Type),
		}

	case model.ToolChoiceRequired:
		if requiredToolChoiceRejected[modelID] {
			prepared.warnings = append(prepared.warnings, model.UnsupportedWarning(
				fmt.Sprintf("tool choice \"required\" for model \"%s\"", modelID),
				"Moonshot AI rejects required tool choice for this model. The setting has been omitted; use \"auto\" or select a specific tool instead.",
			))

			return prepared, nil
		}

		prepared.toolChoice = &toolChoice{
			mode: string(choice.Type),
		}

	case model.ToolChoiceTool:
		prepared.toolChoice = &toolChoice{
			function: choice.ToolName,
		}

	default:
		return preparedTools{}, fmt.Errorf("%w: tool choice type %q", model.ErrUnsupported, choice.Type)
	}

	return prepared, nil
}

// functionToolSpec is a function tool from the call options or from a
// dynamic tool message.
type functionToolSpec struct {
	name        string
	description string
	inputSchema json.RawMessage
	strict      *bool
}

func convertFunctionTool(spec functionToolSpec) (functionTool, error) {
	parameters, err := normalizeSchema(spec.inputSchema)
	if err != nil {
		return functionTool{}, err
	}

	return functionTool{
		Type: functionType,
		Function: functionDefinition{
			Name:        spec.name,
			Description: spec.description,
			Parameters:  parameters,
			Strict:      spec.strict,
		},
	}, nil
}

// errSchemaRoot reports a schema that Moonshot's MFJS validator rejects at
// the root.
var errSchemaRoot = errors.New(`tool parameters must be a JSON Schema object with type "object" for moonshotai (MFJS)`)

// Schema keywords whose values are subschemas.
var (
	schemaArrayKeys = []string{
		"allOf",
		"anyOf",
		"oneOf",
		"prefixItems",
	}

	schemaMapKeys = []string{
		"properties",
		"patternProperties",
		"$defs",
		"dependentSchemas",
	}

	schemaSingleKeys = []string{
		"additionalProperties",
		"propertyNames",
		"items",
		"contains",
		"not",
		"if",
		"then",
		"else",
	}
)

// normalizeSchema converts a JSON Schema to the subset that Moonshot's MFJS
// validator accepts: the root must be an object schema, tuple items become
// prefixItems, and a type next to anyOf moves into the branches. Everything
// else passes through. The original schema is still used to validate
// results.
func normalizeSchema(schema json.RawMessage) (json.RawMessage, error) {
	normalized, err := normalizeDefinition(schema, true)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", model.ErrUnsupported, err)
	}

	return normalized, nil
}

func normalizeDefinition(definition json.RawMessage, isRoot bool) (json.RawMessage, error) {
	object, err := jsonobject.Parse(definition)
	if err != nil {
		if isRoot {
			return nil, errSchemaRoot
		}

		return definition, nil
	}

	if isRoot {
		schemaType, ok := object.GetString("type")
		if !ok || schemaType != "object" {
			return nil, errSchemaRoot
		}
	}

	object, err = normalizeItems(object)
	if err != nil {
		return nil, err
	}

	object, err = moveTypeIntoAnyOf(object)
	if err != nil {
		return nil, err
	}

	for _, key := range schemaArrayKeys {
		object, err = normalizeArrayMember(object, key)
		if err != nil {
			return nil, err
		}
	}

	for _, key := range schemaMapKeys {
		object, err = normalizeMapMember(object, key)
		if err != nil {
			return nil, err
		}
	}

	for _, key := range schemaSingleKeys {
		object, err = normalizeSingleMember(object, key)
		if err != nil {
			return nil, err
		}
	}

	return object.Encode()
}

// normalizeItems appends tuple items to prefixItems, after any existing
// positional items, and normalizes a single items schema.
func normalizeItems(object jsonobject.Object) (jsonobject.Object, error) {
	items, ok := object.Get("items")
	if !ok {
		return object, nil
	}

	tuple, isArray, err := decodeArray(items)
	if err != nil {
		return nil, err
	}

	if !isArray {
		return normalizeSingleMember(object, "items")
	}

	var prefixItems []json.RawMessage
	if existing, ok := object.Get("prefixItems"); ok {
		existingItems, existingIsArray, err := decodeArray(existing)
		if err != nil {
			return nil, err
		}

		if existingIsArray {
			prefixItems = existingItems
		}
	}

	for _, item := range tuple {
		normalized, err := normalizeDefinition(item, false)
		if err != nil {
			return nil, err
		}

		prefixItems = append(prefixItems, normalized)
	}

	encoded, err := encodeArray(prefixItems)
	if err != nil {
		return nil, err
	}

	return object.With("prefixItems", encoded).Without("items"), nil
}

// moveTypeIntoAnyOf moves a string type next to anyOf into the branches
// that have no type, because MFJS requires the type inside the branches.
func moveTypeIntoAnyOf(object jsonobject.Object) (jsonobject.Object, error) {
	parentType, hasType := object.Get("type")
	if !hasType || !isJSONString(parentType) {
		return object, nil
	}

	anyOf, ok := object.Get("anyOf")
	if !ok {
		return object, nil
	}

	branches, isArray, err := decodeArray(anyOf)
	if err != nil || !isArray {
		return object, err
	}

	for index, branch := range branches {
		branchObject, err := jsonobject.Parse(branch)
		if err != nil {
			continue
		}

		if branchType, ok := branchObject.Get("type"); ok && !isJSONNull(branchType) {
			continue
		}

		typedBranch, err := branchObject.WithFirst("type", parentType).Encode()
		if err != nil {
			return nil, err
		}

		branches[index] = typedBranch
	}

	encoded, err := encodeArray(branches)
	if err != nil {
		return nil, err
	}

	return object.Without("type").With("anyOf", encoded), nil
}

func normalizeArrayMember(object jsonobject.Object, key string) (jsonobject.Object, error) {
	value, ok := object.Get(key)
	if !ok {
		return object, nil
	}

	items, isArray, err := decodeArray(value)
	if err != nil || !isArray {
		return object, err
	}

	for index, item := range items {
		normalized, err := normalizeDefinition(item, false)
		if err != nil {
			return nil, err
		}

		items[index] = normalized
	}

	encoded, err := encodeArray(items)
	if err != nil {
		return nil, err
	}

	return object.With(key, encoded), nil
}

func normalizeMapMember(object jsonobject.Object, key string) (jsonobject.Object, error) {
	value, ok := object.Get(key)
	if !ok {
		return object, nil
	}

	members, err := jsonobject.Parse(value)
	if err != nil {
		return object, nil
	}

	for index, member := range members {
		normalized, err := normalizeDefinition(member.Value, false)
		if err != nil {
			return nil, err
		}

		members[index].Value = normalized
	}

	encoded, err := members.Encode()
	if err != nil {
		return nil, err
	}

	return object.With(key, encoded), nil
}

func normalizeSingleMember(object jsonobject.Object, key string) (jsonobject.Object, error) {
	value, ok := object.Get(key)
	if !ok || !jsonobject.IsObject(value) {
		return object, nil
	}

	normalized, err := normalizeDefinition(value, false)
	if err != nil {
		return nil, err
	}

	return object.With(key, normalized), nil
}

// decodeArray decodes value when it is a JSON array and reports whether it
// was one.
func decodeArray(value json.RawMessage) ([]json.RawMessage, bool, error) {
	if !bytes.HasPrefix(bytes.TrimSpace(value), []byte("[")) {
		return nil, false, nil
	}

	var items []json.RawMessage
	if err := json.Unmarshal(value, &items); err != nil {
		return nil, false, fmt.Errorf("decode schema array: %w", err)
	}

	return items, true, nil
}

func encodeArray(items []json.RawMessage) (json.RawMessage, error) {
	if items == nil {
		items = []json.RawMessage{}
	}

	encoded, err := json.Marshal(items)
	if err != nil {
		return nil, fmt.Errorf("encode schema array: %w", err)
	}

	return encoded, nil
}

func isJSONString(value json.RawMessage) bool {
	return bytes.HasPrefix(bytes.TrimSpace(value), []byte(`"`))
}

func isJSONNull(value json.RawMessage) bool {
	return bytes.Equal(bytes.TrimSpace(value), []byte("null"))
}
