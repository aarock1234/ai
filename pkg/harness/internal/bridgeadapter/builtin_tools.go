package bridgeadapter

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/harness"
)

// builtinToolDefinition is one entry of an adapter's built-in tool table.
type builtinToolDefinition struct {
	// Name is what the bridge reports on tool calls: the common name when
	// set, the native name otherwise.
	Name        string                     `json:"name"`
	CommonName  harness.BuiltinToolName    `json:"commonName"`
	NativeName  string                     `json:"nativeName"`
	UseKind     harness.BuiltinToolUseKind `json:"useKind"`
	Description string                     `json:"description"`

	// InputSchema may be omitted for a common tool, which then uses the
	// common baseline schema and description.
	InputSchema json.RawMessage `json:"inputSchema"`
}

// ParseBuiltinTools decodes an adapter's built-in tool table, a JSON array
// of tool definitions in declaration order. Common tools are checked against
// their baseline schemas.
func ParseBuiltinTools(table []byte) ([]harness.BuiltinTool, error) {
	var definitions []builtinToolDefinition
	if err := json.Unmarshal(table, &definitions); err != nil {
		return nil, fmt.Errorf("decode built-in tools: %w", err)
	}

	tools := make([]harness.BuiltinTool, 0, len(definitions))

	for _, definition := range definitions {
		tool, err := builtinToolOf(definition)
		if err != nil {
			return nil, err
		}

		tools = append(tools, tool)
	}

	return tools, nil
}

// builtinToolOf converts one definition.
func builtinToolOf(definition builtinToolDefinition) (harness.BuiltinTool, error) {
	if definition.CommonName == "" {
		return harness.BuiltinTool{
			Tool: ai.Tool{
				Name:        definition.Name,
				Description: definition.Description,
				InputSchema: definition.InputSchema,
			},
			UseKind: definition.UseKind,
		}, nil
	}

	if len(definition.InputSchema) == 0 {
		baseline, ok := commonTool(definition.CommonName)
		if !ok {
			return harness.BuiltinTool{}, fmt.Errorf("built-in tool %q: unknown common name", definition.Name)
		}

		definition.Description = baseline.Description
		definition.InputSchema = baseline.InputSchema
	}

	return harness.CommonTool(definition.CommonName, harness.CommonToolOptions{
		NativeName:  definition.NativeName,
		UseKind:     definition.UseKind,
		Description: definition.Description,
		InputSchema: definition.InputSchema,
	})
}

// commonTool returns the baseline of a common tool.
func commonTool(name harness.BuiltinToolName) (ai.Tool, bool) {
	for _, tool := range harness.CommonTools() {
		if tool.Name == string(name) {
			return tool, true
		}
	}

	return ai.Tool{}, false
}
