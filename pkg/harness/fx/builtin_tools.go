package fx

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/harness"
)

// toolDefinition declares one fx built-in tool.
type toolDefinition struct {
	// name is the reported tool name: the common name of a common tool,
	// and fx's native name otherwise.
	name        string
	nativeName  string
	description string
	useKind     harness.BuiltinToolUseKind

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
	integerSchema = json.RawMessage(`{"type":"integer"}`)
	booleanSchema = json.RawMessage(`{"type":"boolean"}`)
	objectSchema  = json.RawMessage(`{"type":"object"}`)
	arraySchema   = json.RawMessage(`{"type":"array"}`)

	stringArraySchema = json.RawMessage(`{"type":"array","items":{"type":"string"}}`)
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

// toolDefinitions are fx's built-in tools in fx's registry order, as
// declared by src/builtins/tools.zig of the fx v0.0.14 sources. fx reports
// each ACP tool call with its native name, which the bridge maps to these
// tools. Nested request unions are declared as plain objects, and input
// schemas accept additional properties. Upstream instead registers its own
// read, write, edit, bash, grep, glob, ls and askUserQuestions tools with
// libfx; fx's ACP server runs its native tools, and its ask_user_question
// tool answers only in fx's interactive shell.
var toolDefinitions = []toolDefinition{
	{
		name:        "glob",
		nativeName:  "glob_files",
		description: "Find files matching a glob pattern.",
		useKind:     harness.UseReadOnly,
		common:      true,
		properties: []schemaProperty{
			required("pattern", stringSchema),
			optional("path", stringSchema),
			optional("mode", stringSchema),
		},
	},
	{
		name:        "grep",
		nativeName:  "grep_files",
		description: "Search file contents for literal text.",
		useKind:     harness.UseReadOnly,
		common:      true,
		properties: []schemaProperty{
			required("pattern", stringSchema),
			optional("path", stringSchema),
			optional("include", stringSchema),
			optional("case_insensitive", booleanSchema),
			optional("mode", stringSchema),
			optional("head_limit", integerSchema),
			optional("offset", integerSchema),
			optional("context_lines", integerSchema),
		},
	},
	{
		name:        "read_file",
		description: "Read a file.",
		useKind:     harness.UseReadOnly,
		properties: []schemaProperty{
			required("path", stringSchema),
			optional("start_line", integerSchema),
			optional("line_count", integerSchema),
		},
	},
	{
		name:        "write_file",
		description: "Create or overwrite a file.",
		useKind:     harness.UseEdit,
		properties: []schemaProperty{
			required("path", stringSchema),
			required("content", stringSchema),
		},
	},
	{
		name:        "edit_file",
		description: "Replace an exact, unique string in a file.",
		useKind:     harness.UseEdit,
		properties: []schemaProperty{
			required("path", stringSchema),
			required("old_string", stringSchema),
			required("new_string", stringSchema),
		},
	},
	{
		name:        "web_fetch",
		description: "Read a public URL.",
		useKind:     harness.UseReadOnly,
		properties: []schemaProperty{
			required("url", stringSchema),
		},
	},
	{
		name:        "webSearch",
		nativeName:  "web_search",
		description: "Search the web.",
		useKind:     harness.UseReadOnly,
		common:      true,
		properties: []schemaProperty{
			required("query", stringSchema),
			optional("allowed_domains", stringArraySchema),
			optional("blocked_domains", stringArraySchema),
		},
	},
	{
		name:        "shell",
		description: "Run, interact with or stop a shell command.",
		useKind:     harness.UseBash,
		properties: []schemaProperty{
			required("request", objectSchema),
		},
	},
	{
		name:        "capability_search",
		description: "Find installed skills and configured MCP tools.",
		useKind:     harness.UseReadOnly,
		properties: []schemaProperty{
			required("query", stringSchema),
			optional("server", stringSchema),
		},
	},
	{
		name:        "skill",
		description: "Load an installed skill or one of its resources.",
		useKind:     harness.UseReadOnly,
		properties: []schemaProperty{
			required("location", stringSchema),
			optional("resource", stringSchema),
		},
	},
	{
		name:        "install_skill",
		description: "Install a skill into fx's managed skill storage.",
		useKind:     harness.UseEdit,
		properties: []schemaProperty{
			required("source", stringSchema),
			optional("skill", stringSchema),
		},
	},
	{
		name:        "subagent",
		description: "Delegate work to another agent.",
		properties: []schemaProperty{
			required("request", objectSchema),
		},
	},
	{
		name:        "mcp_select_tool",
		description: "Load a specific MCP tool by name.",
		useKind:     harness.UseReadOnly,
		properties: []schemaProperty{
			required("name", stringSchema),
		},
	},
	{
		name:        "mcp_features",
		description: "Read MCP resources and use server prompts.",
		useKind:     harness.UseReadOnly,
		properties: []schemaProperty{
			required("action", stringSchema),
			required("server", stringSchema),
			optional("uri", stringSchema),
			optional("uri_template", stringSchema),
			optional("prompt", stringSchema),
			optional("argument", stringSchema),
			optional("value", stringSchema),
			optional("arguments", objectSchema),
			optional("context", objectSchema),
		},
	},
	{
		name:        "ask_user_question",
		description: "Ask the user multiple-choice questions; unavailable outside fx's interactive shell.",
		useKind:     harness.UseReadOnly,
		properties: []schemaProperty{
			required("questions", arraySchema),
		},
	},
	{
		name:        "vision",
		description: "Analyze images for models without image input.",
		useKind:     harness.UseReadOnly,
		properties: []schemaProperty{
			required("focus", stringSchema),
			optional("image_ids", arraySchema),
			optional("paths", stringArraySchema),
		},
	},
	{
		name:        "read_tool_result",
		description: "Search or read a stored tool result.",
		useKind:     harness.UseReadOnly,
		properties: []schemaProperty{
			required("request", objectSchema),
		},
	},
}

// builtinTools declares fx's built-in tools in fx's registry order.
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
				Description: d.description,
				InputSchema: inputSchema,
			},
			UseKind: d.useKind,
		}, nil
	}

	tool, err := harness.CommonTool(harness.BuiltinToolName(d.name), harness.CommonToolOptions{
		NativeName:  d.nativeName,
		UseKind:     d.useKind,
		Description: d.description,
		InputSchema: inputSchema,
	})
	if err != nil {
		return harness.BuiltinTool{}, fmt.Errorf("declare fx tool %q: %w", d.name, err)
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
		return nil, fmt.Errorf("encode fx tool %q input schema: %w", d.name, err)
	}

	return encoded, nil
}
