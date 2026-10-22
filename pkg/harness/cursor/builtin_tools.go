package cursor

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/harness"
)

// toolDefinition declares one Cursor built-in tool.
type toolDefinition struct {
	name       string
	title      string
	nativeName string
	useKind    harness.BuiltinToolUseKind

	// common declares the tool under its common name.
	common bool

	properties []schemaProperty
}

// schemaProperty is one property of a tool's input schema.
type schemaProperty struct {
	name     string
	schema   json.RawMessage
	required bool
}

// Property schemas.
var (
	stringSchema  = json.RawMessage(`{"type":"string"}`)
	numberSchema  = json.RawMessage(`{"type":"number"}`)
	unknownSchema = json.RawMessage(`{}`)

	stringArraySchema = json.RawMessage(`{"type":"array","items":{"type":"string"}}`)

	todoArraySchema = json.RawMessage(`{
		"type": "array",
		"items": {
			"type": "object",
			"properties": {
				"id": {"type": "string"},
				"content": {"type": "string"},
				"status": {"type": "string"}
			},
			"required": ["id", "content", "status"]
		}
	}`)
)

// required declares a required property.
func required(name string, schema json.RawMessage) schemaProperty {
	return schemaProperty{
		name:     name,
		schema:   schema,
		required: true,
	}
}

// optional declares an optional property.
func optional(name string, schema json.RawMessage) schemaProperty {
	return schemaProperty{
		name:   name,
		schema: schema,
	}
}

// toolNameLiteral declares the required _toolName discriminator Cursor
// sends for tools without a distinctive input.
func toolNameLiteral(name string) schemaProperty {
	return required("_toolName", json.RawMessage(fmt.Sprintf(`{"type":"string","const":%q}`, name)))
}

// toolDefinitions are Cursor's built-in tools. Tool variants, display
// titles, ACP kinds and raw input projections were captured upstream from
// the official Cursor CLI 2026.08.11-e8db854 ACP implementation.
// mcpToolCall is dynamic and truncatedToolCall is a transport sentinel, so
// neither is a built-in tool. Input schemas accept additional properties.
var toolDefinitions = []toolDefinition{
	{
		name:       "bash",
		title:      "Terminal",
		nativeName: "shellToolCall",
		useKind:    harness.UseBash,
		common:     true,
		properties: []schemaProperty{
			required("command", stringSchema),
		},
	},
	{
		name:       "delete",
		title:      "Delete",
		nativeName: "deleteToolCall",
		useKind:    harness.UseEdit,
		properties: []schemaProperty{
			required("path", stringSchema),
		},
	},
	{
		name:       "glob",
		title:      "Find",
		nativeName: "globToolCall",
		useKind:    harness.UseReadOnly,
		common:     true,
		properties: []schemaProperty{
			required("pattern", stringSchema),
		},
	},
	{
		name:       "grep",
		title:      "grep",
		nativeName: "grepToolCall",
		useKind:    harness.UseReadOnly,
		common:     true,
		properties: []schemaProperty{
			required("pattern", stringSchema),
			optional("path", stringSchema),
		},
	},
	{
		name:       "read",
		title:      "Read",
		nativeName: "readToolCall",
		useKind:    harness.UseReadOnly,
		properties: []schemaProperty{
			required("path", stringSchema),
		},
	},
	{
		name:       "updateTodos",
		title:      "Update TODOs",
		nativeName: "updateTodosToolCall",
		properties: []schemaProperty{
			toolNameLiteral("updateTodos"),
			optional("todos", todoArraySchema),
		},
	},
	{
		name:       "readTodos",
		title:      "Read TODOs",
		nativeName: "readTodosToolCall",
		useKind:    harness.UseReadOnly,
	},
	{
		name:       "edit",
		title:      "Edit",
		nativeName: "editToolCall",
		useKind:    harness.UseEdit,
		properties: []schemaProperty{
			required("path", stringSchema),
		},
	},
	{
		name:       "ls",
		title:      "List",
		nativeName: "lsToolCall",
		useKind:    harness.UseReadOnly,
		properties: []schemaProperty{
			required("path", stringSchema),
		},
	},
	{
		name:       "readLints",
		title:      "Read Lints",
		nativeName: "readLintsToolCall",
		useKind:    harness.UseReadOnly,
		properties: []schemaProperty{
			required("paths", stringArraySchema),
		},
	},
	{
		name:       "semanticSearch",
		title:      "Codebase Search",
		nativeName: "semSearchToolCall",
		useKind:    harness.UseReadOnly,
		properties: []schemaProperty{
			required("query", stringSchema),
		},
	},
	{
		name:       "createPlan",
		title:      "Create Plan",
		nativeName: "createPlanToolCall",
		properties: []schemaProperty{
			toolNameLiteral("createPlan"),
			optional("name", stringSchema),
			optional("plan", stringSchema),
		},
	},
	{
		name:       "webSearch",
		title:      "Web Search",
		nativeName: "webSearchToolCall",
		useKind:    harness.UseReadOnly,
		properties: []schemaProperty{
			required("searchTerm", stringSchema),
		},
	},
	{
		name:       "task",
		title:      "Task",
		nativeName: "taskToolCall",
		properties: []schemaProperty{
			toolNameLiteral("task"),
			optional("prompt", stringSchema),
			optional("description", stringSchema),
			optional("subagentType", unknownSchema),
		},
	},
	{
		name:       "listMcpResources",
		title:      "List MCP Resources",
		nativeName: "listMcpResourcesToolCall",
		useKind:    harness.UseReadOnly,
		properties: []schemaProperty{
			optional("server", stringSchema),
		},
	},
	{
		name:       "readMcpResource",
		title:      "Fetch MCP Resource",
		nativeName: "readMcpResourceToolCall",
		useKind:    harness.UseReadOnly,
		properties: []schemaProperty{
			optional("server", stringSchema),
			optional("uri", stringSchema),
			optional("downloadPath", stringSchema),
		},
	},
	{
		name:       "applyAgentDiff",
		title:      "Apply Agent Diff",
		nativeName: "applyAgentDiffToolCall",
		useKind:    harness.UseEdit,
		properties: []schemaProperty{
			optional("path", stringSchema),
		},
	},
	{
		name:       "fetch",
		title:      "Fetch",
		nativeName: "fetchToolCall",
		useKind:    harness.UseReadOnly,
		properties: []schemaProperty{
			required("url", stringSchema),
		},
	},
	{
		name:       "switchMode",
		title:      "Switch Mode",
		nativeName: "switchModeToolCall",
		properties: []schemaProperty{
			required("targetModeId", stringSchema),
			optional("explanation", stringSchema),
		},
	},
	{
		name:       "generateImage",
		title:      "Generate Image",
		nativeName: "generateImageToolCall",
		properties: []schemaProperty{
			toolNameLiteral("generateImage"),
			optional("description", stringSchema),
			optional("filename", stringSchema),
		},
	},
	{
		name:       "recordScreen",
		title:      "Record Screen",
		nativeName: "recordScreenToolCall",
	},
	{
		name:       "computerUse",
		title:      "Computer Use",
		nativeName: "computerUseToolCall",
		useKind:    harness.UseBash,
		properties: []schemaProperty{
			optional("action", stringSchema),
			optional("coordinate", unknownSchema),
			optional("text", stringSchema),
		},
	},
	{
		name:       "writeShellStdin",
		title:      "Write to stdin",
		nativeName: "writeShellStdinToolCall",
		useKind:    harness.UseBash,
		properties: []schemaProperty{
			optional("shellId", numberSchema),
			optional("chars", stringSchema),
		},
	},
	{
		name:       "reflect",
		title:      "Reflect",
		nativeName: "reflectToolCall",
		useKind:    harness.UseReadOnly,
		properties: []schemaProperty{
			optional("reflection", stringSchema),
		},
	},
	{
		name:       "setupVmEnvironment",
		title:      "Setup VM Environment",
		nativeName: "setupVmEnvironmentToolCall",
		useKind:    harness.UseBash,
	},
	{
		name:       "replaceEnv",
		title:      "Replace Environment",
		nativeName: "replaceEnvToolCall",
		useKind:    harness.UseBash,
	},
	{
		name:       "startGrindExecution",
		title:      "Start Grind Execution",
		nativeName: "startGrindExecutionToolCall",
		useKind:    harness.UseBash,
	},
	{
		name:       "startGrindPlanning",
		title:      "Start Grind Planning",
		nativeName: "startGrindPlanningToolCall",
		useKind:    harness.UseReadOnly,
	},
	{
		name:       "webFetch",
		title:      "Web Fetch",
		nativeName: "webFetchToolCall",
		useKind:    harness.UseReadOnly,
		properties: []schemaProperty{
			required("url", stringSchema),
		},
	},
	{
		name:       "reportBugfixResults",
		title:      "Report Bugfix Results",
		nativeName: "reportBugfixResultsToolCall",
	},
}

// builtinTools declares Cursor's built-in tools in upstream order.
func builtinTools() ([]harness.BuiltinTool, error) {
	tools := make([]harness.BuiltinTool, 0, len(toolDefinitions))

	for _, definition := range toolDefinitions {
		tool, err := definition.builtinTool()
		if err != nil {
			return nil, err
		}

		tools = append(tools, tool)
	}

	return tools, nil
}

// builtinTool declares one tool, checking common tools against their
// baseline schema.
func (d toolDefinition) builtinTool() (harness.BuiltinTool, error) {
	inputSchema, err := d.inputSchema()
	if err != nil {
		return harness.BuiltinTool{}, err
	}

	if !d.common {
		return harness.BuiltinTool{
			Name:        d.name,
			Title:       d.title,
			InputSchema: inputSchema,
			NativeName:  d.nativeName,
			UseKind:     d.useKind,
		}, nil
	}

	tool, err := harness.CommonTool(harness.BuiltinToolName(d.name), harness.CommonToolOptions{
		NativeName:  d.nativeName,
		UseKind:     d.useKind,
		InputSchema: inputSchema,
	})
	if err != nil {
		return harness.BuiltinTool{}, fmt.Errorf("declare cursor tool %q: %w", d.name, err)
	}

	tool.Title = d.title

	return tool, nil
}

// inputSchema returns the object schema of the tool's properties.
func (d toolDefinition) inputSchema() (json.RawMessage, error) {
	properties := make(map[string]json.RawMessage, len(d.properties))
	requiredNames := []string{}

	for _, property := range d.properties {
		properties[property.name] = property.schema

		if property.required {
			requiredNames = append(requiredNames, property.name)
		}
	}

	schema := struct {
		Type       string                     `json:"type"`
		Properties map[string]json.RawMessage `json:"properties"`
		Required   []string                   `json:"required,omitempty"`
	}{
		Type:       "object",
		Properties: properties,
		Required:   requiredNames,
	}

	encoded, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("encode cursor tool %q input schema: %w", d.name, err)
	}

	return encoded, nil
}
