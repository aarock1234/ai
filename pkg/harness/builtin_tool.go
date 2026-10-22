package harness

import (
	"encoding/json"
	"fmt"
	"slices"

	"github.com/aarock1234/ai/pkg/ai"
)

// BuiltinTool is a tool the runtime exposes natively. Its Name is what the
// adapter reports on tool calls: the common name when the tool maps to one,
// and the native name otherwise. Built-in tools are executed by the runtime,
// so Execute is normally nil.
type BuiltinTool struct {
	ai.Tool

	// NativeName is the runtime's name of the tool when it differs from
	// Name, for example "Bash" for the common "bash" tool.
	NativeName string

	// CommonName is the cross-harness name of the capability, so that
	// consumers recognize equivalent tools of different runtimes. It is empty
	// for tools without a common equivalent.
	CommonName BuiltinToolName

	// UseKind classifies the tool for permission modes.
	UseKind BuiltinToolUseKind
}

// BuiltinToolName is a cross-harness name of a common built-in tool.
type BuiltinToolName string

// Common built-in tool names.
const (
	ToolRead             BuiltinToolName = "read"
	ToolWrite            BuiltinToolName = "write"
	ToolEdit             BuiltinToolName = "edit"
	ToolBash             BuiltinToolName = "bash"
	ToolGrep             BuiltinToolName = "grep"
	ToolGlob             BuiltinToolName = "glob"
	ToolWebSearch        BuiltinToolName = "webSearch"
	ToolAskUserQuestions BuiltinToolName = "askUserQuestions"
)

// BuiltinToolUseKind classifies a built-in tool for permission modes.
type BuiltinToolUseKind string

// Built-in tool use kinds.
const (
	UseReadOnly BuiltinToolUseKind = "readonly"
	UseEdit     BuiltinToolUseKind = "edit"
	UseBash     BuiltinToolUseKind = "bash"
)

// commonToolDefinition is the baseline of one common built-in tool.
type commonToolDefinition struct {
	name        BuiltinToolName
	description string
	inputSchema string

	// outputSchema is empty for tools whose output is unknown.
	outputSchema string
}

// commonToolDefinitions is the cross-harness vocabulary of common built-in
// tools with their baseline input schemas, in upstream order.
var commonToolDefinitions = []commonToolDefinition{
	{
		name:        ToolRead,
		description: "Read file contents",
		inputSchema: stringPropertiesSchema("file_path"),
	},
	{
		name:        ToolWrite,
		description: "Write content to a file",
		inputSchema: stringPropertiesSchema("file_path", "content"),
	},
	{
		name:        ToolEdit,
		description: "Edit a file by replacing text",
		inputSchema: stringPropertiesSchema("file_path", "old_string", "new_string"),
	},
	{
		name:        ToolBash,
		description: "Execute a shell command",
		inputSchema: stringPropertiesSchema("command"),
	},
	{
		name:        ToolGrep,
		description: "Search file contents with regex",
		inputSchema: stringPropertiesSchema("pattern"),
	},
	{
		name:        ToolGlob,
		description: "Find files matching a glob pattern",
		inputSchema: stringPropertiesSchema("pattern"),
	},
	{
		name:        ToolWebSearch,
		description: "Search the web",
		inputSchema: stringPropertiesSchema("query"),
	},
	{
		name:         ToolAskUserQuestions,
		description:  questionsToolDescription,
		inputSchema:  QuestionsToolInputSchema,
		outputSchema: QuestionsToolOutputSchema,
	},
}

// CommonTools returns the common built-in tools with their baseline input
// schemas, and output schemas where the output is known, in the order of
// the upstream vocabulary. Adapters that declare a tool with one of these
// names must accept every input the baseline accepts.
func CommonTools() []ai.Tool {
	tools := make([]ai.Tool, 0, len(commonToolDefinitions))

	for _, definition := range commonToolDefinitions {
		tool := ai.Tool{
			Name:        string(definition.name),
			Description: definition.description,
			InputSchema: json.RawMessage(definition.inputSchema),
		}

		if definition.outputSchema != "" {
			tool.OutputSchema = json.RawMessage(definition.outputSchema)
		}

		tools = append(tools, tool)
	}

	return tools
}

// CommonToolOptions configure CommonTool.
type CommonToolOptions struct {
	// NativeName is the runtime's name of the tool. It is required.
	NativeName string

	UseKind     BuiltinToolUseKind
	Description string

	// InputSchema is the adapter's input schema. It must accept every input
	// the baseline schema of the common name accepts; extra optional fields
	// are encouraged.
	InputSchema json.RawMessage
}

// CommonTool declares a built-in tool that maps to a common name.
//
// Upstream checks at compile time that the adapter's schema is a superset of
// the baseline; Go checks it when the tool is declared: every property the
// baseline defines must be declared with a compatible type, and the adapter
// must not require properties the baseline does not require.
func CommonTool(name BuiltinToolName, options CommonToolOptions) (BuiltinTool, error) {
	baseline, ok := commonToolBaseline(name)
	if !ok {
		return BuiltinTool{}, fmt.Errorf("declare common tool: unknown common name %q", name)
	}

	if options.NativeName == "" {
		return BuiltinTool{}, fmt.Errorf("declare common tool %q: native name is required", name)
	}

	if err := checkSchemaSuperset(json.RawMessage(baseline.inputSchema), options.InputSchema); err != nil {
		return BuiltinTool{}, fmt.Errorf("declare common tool %q: %w", name, err)
	}

	return BuiltinTool{
		Tool: ai.Tool{
			Name:        string(name),
			Description: options.Description,
			InputSchema: options.InputSchema,
		},
		NativeName: options.NativeName,
		CommonName: name,
		UseKind:    options.UseKind,
	}, nil
}

// commonToolBaseline returns the baseline definition of a common name.
func commonToolBaseline(name BuiltinToolName) (commonToolDefinition, bool) {
	for _, definition := range commonToolDefinitions {
		if definition.name == name {
			return definition, true
		}
	}

	return commonToolDefinition{}, false
}

// objectSchema is the part of an object JSON Schema the superset check
// reads.
type objectSchema struct {
	Properties           map[string]propertySchema `json:"properties"`
	Required             []string                  `json:"required"`
	AdditionalProperties *bool                     `json:"additionalProperties"`
}

// propertySchema is the part of a property schema the superset check reads.
type propertySchema struct {
	Type json.RawMessage `json:"type"`
}

// checkSchemaSuperset reports an error when the adapter schema may reject an
// input the baseline accepts.
func checkSchemaSuperset(baselineSchema json.RawMessage, adapterSchema json.RawMessage) error {
	var baseline objectSchema
	if err := json.Unmarshal(baselineSchema, &baseline); err != nil {
		return fmt.Errorf("decode baseline schema: %w", err)
	}

	var adapter objectSchema
	if err := json.Unmarshal(adapterSchema, &adapter); err != nil {
		return fmt.Errorf("decode input schema: %w", err)
	}

	for _, required := range adapter.Required {
		if !slices.Contains(baseline.Required, required) {
			return fmt.Errorf("input schema requires %q, which the baseline schema does not require", required)
		}
	}

	for name, baselineProperty := range baseline.Properties {
		adapterProperty, declared := adapter.Properties[name]
		if !declared {
			if adapter.AdditionalProperties != nil && !*adapter.AdditionalProperties {
				return fmt.Errorf("input schema rejects the baseline property %q", name)
			}

			continue
		}

		if !compatiblePropertyType(baselineProperty.Type, adapterProperty.Type) {
			return fmt.Errorf("input schema declares the baseline property %q with an incompatible type", name)
		}
	}

	return nil
}

// compatiblePropertyType reports whether the adapter's property type accepts
// the baseline's. An adapter property without a type accepts anything.
func compatiblePropertyType(baselineType json.RawMessage, adapterType json.RawMessage) bool {
	if len(adapterType) == 0 || len(baselineType) == 0 {
		return true
	}

	baselineTypes := schemaTypes(baselineType)
	adapterTypes := schemaTypes(adapterType)

	for _, required := range baselineTypes {
		if !slices.Contains(adapterTypes, required) {
			return false
		}
	}

	return true
}

// schemaTypes decodes a JSON Schema type keyword, a string or a list.
func schemaTypes(raw json.RawMessage) []string {
	var single string
	if err := json.Unmarshal(raw, &single); err == nil {
		return []string{
			single,
		}
	}

	var several []string
	if err := json.Unmarshal(raw, &several); err == nil {
		return several
	}

	return nil
}

// stringPropertiesSchema returns an object schema requiring the named string
// properties.
func stringPropertiesSchema(names ...string) string {
	properties := make(map[string]map[string]string, len(names))
	for _, name := range names {
		properties[name] = map[string]string{
			"type": "string",
		}
	}

	schema, err := json.Marshal(map[string]any{
		"type":                 "object",
		"properties":           properties,
		"required":             names,
		"additionalProperties": false,
	})
	if err != nil {
		panic(fmt.Sprintf("encode common tool schema: %v", err))
	}

	return string(schema)
}
