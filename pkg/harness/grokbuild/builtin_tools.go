package grokbuild

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/harness"
)

// toolDefinition declares one Grok Build built-in tool.
type toolDefinition struct {
	name    string
	useKind harness.BuiltinToolUseKind

	// commonName declares the tool under a common name, with name as its
	// native name.
	commonName harness.BuiltinToolName

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
	stringSchema             = json.RawMessage(`{"type":"string"}`)
	integerSchema            = json.RawMessage(`{"type":"integer"}`)
	booleanSchema            = json.RawMessage(`{"type":"boolean"}`)
	objectSchema             = json.RawMessage(`{"type":"object"}`)
	unknownSchema            = json.RawMessage(`{}`)
	stringArraySchema        = json.RawMessage(`{"type":"array","items":{"type":"string"}}`)
	nonNegativeIntegerSchema = json.RawMessage(`{"type":"integer","minimum":0}`)

	todoArraySchema = json.RawMessage(`{
		"type": "array",
		"items": {
			"type": "object",
			"properties": {
				"id": {"type": "string"},
				"content": {"anyOf": [{"type": "string"}, {"type": "null"}]},
				"status": {
					"anyOf": [
						{"type": "string", "enum": ["pending", "in_progress", "completed", "cancelled"]},
						{"type": "null"}
					]
				}
			},
			"required": ["id"]
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

// nullable allows null in place of a schema's values.
func nullable(schema json.RawMessage) json.RawMessage {
	return json.RawMessage(fmt.Sprintf(`{"anyOf":[%s,{"type":"null"}]}`, schema))
}

// boundedInteger returns the schema of an integer between minimum and
// maximum.
func boundedInteger(minimum int, maximum int) json.RawMessage {
	return json.RawMessage(fmt.Sprintf(`{"type":"integer","minimum":%d,"maximum":%d}`, minimum, maximum))
}

// enumSchema returns the schema of a string limited to values.
func enumSchema(values ...string) json.RawMessage {
	schema, err := json.Marshal(struct {
		Type string   `json:"type"`
		Enum []string `json:"enum"`
	}{
		Type: "string",
		Enum: values,
	})
	if err != nil {
		panic(fmt.Sprintf("encode grok build enum schema: %v", err))
	}

	return schema
}

// toolDefinitions are Grok Build's built-in tools in upstream order, with
// the names and inputs upstream captured from the model requests of
// @xai-official/grok 0.2.111. The common askUserQuestions tool, which
// upstream declares first, is added by the acp package. Input schemas
// accept additional properties.
var toolDefinitions = []toolDefinition{
	{
		name:       "run_terminal_command",
		commonName: harness.ToolBash,
		useKind:    harness.UseBash,
		properties: []schemaProperty{
			required("command", stringSchema),
			optional("timeout", nullable(boundedInteger(0, 36_000_000))),
			optional("description", stringSchema),
			optional("background", booleanSchema),
		},
	},
	{
		name:       "search_replace",
		commonName: harness.ToolEdit,
		useKind:    harness.UseEdit,
		properties: []schemaProperty{
			required("file_path", stringSchema),
			required("old_string", stringSchema),
			required("new_string", stringSchema),
			optional("replace_all", booleanSchema),
		},
	},
	{
		name:       "grep",
		commonName: harness.ToolGrep,
		useKind:    harness.UseReadOnly,
		properties: []schemaProperty{
			required("pattern", stringSchema),
			optional("path", nullable(stringSchema)),
			optional("glob", nullable(stringSchema)),
			optional("-B", integerSchema),
			optional("-A", integerSchema),
			optional("-C", integerSchema),
			optional("-i", booleanSchema),
			optional("type", nullable(stringSchema)),
			optional("head_limit", integerSchema),
			optional("multiline", booleanSchema),
		},
	},
	{
		name:       "web_search",
		commonName: harness.ToolWebSearch,
		useKind:    harness.UseReadOnly,
		properties: []schemaProperty{
			required("query", stringSchema),
			optional("allowed_domains", nullable(stringArraySchema)),
		},
	},
	{
		name:       "write",
		commonName: harness.ToolWrite,
		useKind:    harness.UseEdit,
		properties: []schemaProperty{
			required("file_path", stringSchema),
			required("content", stringSchema),
		},
	},
	{
		name:    "read_file",
		useKind: harness.UseReadOnly,
		properties: []schemaProperty{
			required("target_file", stringSchema),
			optional("offset", integerSchema),
			optional("limit", integerSchema),
			optional("pages", nullable(stringSchema)),
			optional("format", nullable(stringSchema)),
		},
	},
	{
		name:    "list_dir",
		useKind: harness.UseReadOnly,
		properties: []schemaProperty{
			required("target_directory", stringSchema),
		},
	},
	{
		name: "kill_command_or_subagent",
		properties: []schemaProperty{
			required("task_id", stringSchema),
		},
	},
	{
		name: "todo_write",
		properties: []schemaProperty{
			optional("merge", booleanSchema),
			required("todos", todoArraySchema),
		},
	},
	{
		name:    "get_command_or_subagent_output",
		useKind: harness.UseReadOnly,
		properties: []schemaProperty{
			optional("task_ids", stringArraySchema),
			optional("timeout_ms", nullable(nonNegativeIntegerSchema)),
		},
	},
	{
		name: "spawn_subagent",
		properties: []schemaProperty{
			required("prompt", stringSchema),
			required("description", stringSchema),
			optional("subagent_type", stringSchema),
			optional("background", booleanSchema),
			optional("capability_mode", nullable(enumSchema("read-only", "read-write", "execute", "all"))),
			optional("isolation", nullable(enumSchema("none", "worktree"))),
			optional("resume_from", nullable(stringSchema)),
			optional("cwd", nullable(stringSchema)),
			optional("model", nullable(stringSchema)),
		},
	},
	{
		name: "scheduler_create",
		properties: []schemaProperty{
			optional("task_id", nullable(stringSchema)),
			optional("interval", nullable(stringSchema)),
			optional("prompt", nullable(stringSchema)),
			optional("durable", nullable(booleanSchema)),
			optional("foreground", nullable(booleanSchema)),
			optional("fire_immediately", booleanSchema),
		},
	},
	{
		name: "scheduler_delete",
		properties: []schemaProperty{
			required("id", stringSchema),
		},
	},
	{
		name:    "scheduler_list",
		useKind: harness.UseReadOnly,
	},
	{
		name: "monitor",
		properties: []schemaProperty{
			required("command", stringSchema),
			required("description", stringSchema),
			optional("timeout_ms", nullable(nonNegativeIntegerSchema)),
			optional("persistent", booleanSchema),
		},
	},
	{
		name:    "search_tool",
		useKind: harness.UseReadOnly,
		properties: []schemaProperty{
			required("query", stringSchema),
			optional("limit", nullable(boundedInteger(0, 255))),
		},
	},
	{
		name: "use_tool",
		properties: []schemaProperty{
			required("tool_name", stringSchema),
			required("tool_input", objectSchema),
		},
	},
	{
		name: "workflow",
		properties: []schemaProperty{
			optional("agent_budget", nullable(boundedInteger(1, 1024))),
			optional("name", nullable(stringSchema)),
			optional("script", nullable(stringSchema)),
			optional("script_path", nullable(stringSchema)),
			optional("args", unknownSchema),
			optional("resume_from_run_id", nullable(stringSchema)),
			optional("validate_only", booleanSchema),
		},
	},
	{
		name: "enter_plan_mode",
	},
	{
		name: "exit_plan_mode",
	},
	{
		name: "image_gen",
		properties: []schemaProperty{
			required("prompt", stringSchema),
			optional("aspect_ratio", stringSchema),
		},
	},
	{
		name: "image_edit",
		properties: []schemaProperty{
			required("prompt", stringSchema),
			required("image", stringArraySchema),
			optional("aspect_ratio", stringSchema),
		},
	},
	{
		name: "image_to_video",
		properties: []schemaProperty{
			optional("prompt", nullable(stringSchema)),
			required("image", stringSchema),
			optional("duration", nullable(nonNegativeIntegerSchema)),
			optional("resolution_name", stringSchema),
		},
	},
	{
		name: "reference_to_video",
		properties: []schemaProperty{
			required("prompt", stringSchema),
			required("images", stringArraySchema),
			required("aspect_ratio", stringSchema),
			optional("duration", nullable(nonNegativeIntegerSchema)),
			optional("resolution_name", stringSchema),
		},
	},
}

// builtinTools declares Grok Build's built-in tools in upstream order.
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

	if d.commonName == "" {
		return harness.BuiltinTool{
			Name:        d.name,
			InputSchema: inputSchema,
			UseKind:     d.useKind,
		}, nil
	}

	tool, err := harness.CommonTool(d.commonName, harness.CommonToolOptions{
		NativeName:  d.name,
		UseKind:     d.useKind,
		InputSchema: inputSchema,
	})
	if err != nil {
		return harness.BuiltinTool{}, fmt.Errorf("declare grok build tool %q: %w", d.name, err)
	}

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
		return nil, fmt.Errorf("encode grok build tool %q input schema: %w", d.name, err)
	}

	return encoded, nil
}
