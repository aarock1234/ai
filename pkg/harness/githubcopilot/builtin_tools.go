package githubcopilot

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/harness"
)

// toolDefinition declares one GitHub Copilot built-in tool.
type toolDefinition struct {
	name    string
	title   string
	useKind harness.BuiltinToolUseKind

	// common declares the tool under its common name, with the tool name as
	// its native name.
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
	stringSchema             = json.RawMessage(`{"type":"string"}`)
	numberSchema             = json.RawMessage(`{"type":"number"}`)
	booleanSchema            = json.RawMessage(`{"type":"boolean"}`)
	nonNegativeIntegerSchema = json.RawMessage(`{"type":"integer","minimum":0}`)
	stringArraySchema        = json.RawMessage(`{"type":"array","items":{"type":"string"}}`)
	numberArraySchema        = json.RawMessage(`{"type":"array","items":{"type":"number"}}`)
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
		panic(fmt.Sprintf("encode github copilot enum schema: %v", err))
	}

	return schema
}

// toolDefinitions are the stable, non-MCP built-in tools GitHub Copilot CLI
// 1.0.82 and later emits, in upstream order. Input schemas accept
// additional properties, so the CLI can add fields without breaking the
// established inputs.
var toolDefinitions = []toolDefinition{
	{
		name:    "bash",
		useKind: harness.UseBash,
		common:  true,
		properties: []schemaProperty{
			required("command", stringSchema),
			optional("description", stringSchema),
			optional("shellId", stringSchema),
			optional("mode", enumSchema("sync", "async")),
			optional("detach", booleanSchema),
			optional("initial_wait", numberSchema),
		},
	},
	{
		name:    "read_bash",
		title:   "Reading shell output",
		useKind: harness.UseReadOnly,
		properties: []schemaProperty{
			required("shellId", stringSchema),
			required("delay", numberSchema),
		},
	},
	{
		name:    "stop_bash",
		title:   "Stopping shell session",
		useKind: harness.UseBash,
		properties: []schemaProperty{
			required("shellId", stringSchema),
		},
	},
	{
		name:    "list_bash",
		title:   "list_bash",
		useKind: harness.UseReadOnly,
	},
	{
		name:    "view",
		title:   "Viewing ",
		useKind: harness.UseReadOnly,
		properties: []schemaProperty{
			required("path", stringSchema),
			optional("view_range", numberArraySchema),
			optional("forceReadLargeFiles", booleanSchema),
		},
	},
	{
		name:    "create",
		title:   "Creating ",
		useKind: harness.UseEdit,
		properties: []schemaProperty{
			required("path", stringSchema),
			required("file_text", stringSchema),
		},
	},
	{
		name:    "edit",
		title:   "Editing ",
		useKind: harness.UseEdit,
		properties: []schemaProperty{
			required("path", stringSchema),
			optional("old_str", stringSchema),
			required("new_str", stringSchema),
		},
	},
	{
		name:    "web_fetch",
		title:   "Fetching ",
		useKind: harness.UseReadOnly,
		properties: []schemaProperty{
			required("url", stringSchema),
			optional("max_length", numberSchema),
			optional("start_index", numberSchema),
			optional("raw", booleanSchema),
		},
	},
	{
		name:  "skill",
		title: "Using skill: ",
		properties: []schemaProperty{
			required("skill", stringSchema),
		},
	},
	{
		name: "sql",
		properties: []schemaProperty{
			required("description", stringSchema),
			required("query", stringSchema),
			optional("database", enumSchema("session", "session_store")),
		},
	},
	{
		name:    "read_agent",
		title:   "read_agent",
		useKind: harness.UseReadOnly,
		properties: []schemaProperty{
			required("agent_id", stringSchema),
			optional("wait", booleanSchema),
			optional("timeout", numberSchema),
			optional("since_turn", nonNegativeIntegerSchema),
		},
	},
	{
		name:    "list_agents",
		title:   "list_agents",
		useKind: harness.UseReadOnly,
		properties: []schemaProperty{
			optional("include_completed", booleanSchema),
			optional("scope", enumSchema("siblings", "children", "all")),
		},
	},
	{
		name:  "write_agent",
		title: "write_agent",
		properties: []schemaProperty{
			required("message", stringSchema),
			optional("agent_id", stringSchema),
			optional("agent_ids", stringArraySchema),
			optional("scope", enumSchema("siblings", "children")),
		},
	},
	{
		name:    "grep",
		title:   "Searching for ",
		useKind: harness.UseReadOnly,
		common:  true,
		properties: []schemaProperty{
			required("pattern", stringSchema),
			optional("path", stringSchema),
			optional("paths", stringArraySchema),
			optional("glob", stringSchema),
			optional("output_mode", enumSchema("content", "files_with_matches", "count")),
			optional("case_insensitive", booleanSchema),
			optional("multiline", booleanSchema),
			optional("head_limit", numberSchema),
			optional("context", numberSchema),
			optional("before_context", numberSchema),
			optional("after_context", numberSchema),
		},
	},
	{
		name:    "glob",
		title:   "Finding files matching ",
		useKind: harness.UseReadOnly,
		common:  true,
		properties: []schemaProperty{
			required("pattern", stringSchema),
			optional("paths", stringArraySchema),
		},
	},
	{
		name: "task",
		properties: []schemaProperty{
			required("name", stringSchema),
			required("prompt", stringSchema),
			required("agent_type", enumSchema(
				"explore",
				"task",
				"general-purpose",
				"code-review",
				"research",
				"security-review",
			)),
			required("description", stringSchema),
			optional("model", stringSchema),
			optional("context_tier", enumSchema("default", "long_context")),
			optional("mode", enumSchema("sync", "background")),
		},
	},
}

// builtinTools declares GitHub Copilot's built-in tools in upstream order.
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
			Tool: ai.Tool{
				Name:        d.name,
				Title:       d.title,
				InputSchema: inputSchema,
			},
			UseKind: d.useKind,
		}, nil
	}

	tool, err := harness.CommonTool(harness.BuiltinToolName(d.name), harness.CommonToolOptions{
		NativeName:  d.name,
		UseKind:     d.useKind,
		InputSchema: inputSchema,
	})
	if err != nil {
		return harness.BuiltinTool{}, fmt.Errorf("declare github copilot tool %q: %w", d.name, err)
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
		return nil, fmt.Errorf("encode github copilot tool %q input schema: %w", d.name, err)
	}

	return encoded, nil
}
