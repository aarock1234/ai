package codemode

import (
	"embed"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"text/template"

	"github.com/aarock1234/ai/pkg/ai"
)

//go:embed prompts/*.md
var promptFiles embed.FS

var prompts = template.Must(template.ParseFS(promptFiles, "prompts/*.md"))

var (
	identifierPattern        = regexp.MustCompile(`^[A-Za-z_$][A-Za-z0-9_$]*$`)
	identifierInvalidPattern = regexp.MustCompile(`[^A-Za-z0-9_$]`)
	identifierLeadingPattern = regexp.MustCompile(`^[^A-Za-z_$]+`)
	commentWhitespacePattern = regexp.MustCompile(`\s+`)
)

const (
	defaultExampleVariable    = "result"
	defaultIdentifierFallback = "tool"
)

// toolDescriptionData fills prompts/tool_description.md.
type toolDescriptionData struct {
	Conversation bool
	Catalog      string
}

// toolCatalogData fills prompts/tool_catalog.md.
type toolCatalogData struct {
	Declarations string
	Examples     string
}

// catalogMessageData fills prompts/catalog_message.md.
type catalogMessageData struct {
	Catalog string
}

// buildToolDescription returns the description of the code mode tool. With
// conversation discovery it is stable and the catalog is announced in a
// user message instead.
func buildToolDescription(tools []ai.Tool, discovery ToolDiscovery) string {
	data := toolDescriptionData{
		Conversation: discovery == ToolDiscoveryConversation,
	}

	if !data.Conversation {
		data.Catalog = renderToolCatalog(tools)
	}

	return renderPrompt("tool_description.md", data)
}

// buildCatalogMessage returns the user message that announces the host
// tools with conversation discovery.
func buildCatalogMessage(tools []ai.Tool) string {
	return renderPrompt("catalog_message.md", catalogMessageData{
		Catalog: renderToolCatalog(tools),
	})
}

// jsInputDescription describes the js input of the code mode tool.
func jsInputDescription() string {
	return renderPrompt("js_input.md", nil)
}

// renderToolCatalog renders the TypeScript declarations of the host tools
// and examples that call them.
func renderToolCatalog(tools []ai.Tool) string {
	data := toolCatalogData{}

	if len(tools) > 0 {
		var declarations []string
		for _, tool := range tools {
			declarations = append(declarations, renderToolDeclaration(tool)...)
		}

		data.Declarations = strings.Join(declarations, "\n")
		data.Examples = strings.Join(renderToolExamples(tools), "\n")
	}

	return renderPrompt("tool_catalog.md", data)
}

// renderPrompt executes an embedded prompt template. The templates are
// compiled into the package and only receive strings and booleans, so a
// failure is a programming error.
func renderPrompt(name string, data any) string {
	var builder strings.Builder

	if err := prompts.ExecuteTemplate(&builder, name, data); err != nil {
		panic(fmt.Sprintf("code mode: render prompt %s: %v", name, err))
	}

	return strings.TrimRight(builder.String(), "\n")
}

// renderToolDeclaration renders the declaration of one tool. Inputs and
// outputs without a schema are unknown.
func renderToolDeclaration(tool ai.Tool) []string {
	inputType := schemaTypeOrUnknown(tool.InputSchema)
	outputType := schemaTypeOrUnknown(tool.OutputSchema)

	var lines []string

	if description := firstNonEmpty(tool.Description, tool.Title); description != "" {
		lines = append(lines, fmt.Sprintf("  /** %s */", toComment(description)))
	}

	lines = append(lines, fmt.Sprintf(
		"  %s: (input: %s) => Promise<%s>;",
		formatObjectKey(tool.Name),
		indentType(inputType, 2),
		indentType(outputType, 2),
	))

	return lines
}

// schemaTypeOrUnknown renders a JSON Schema as a TypeScript type, or
// unknown when the schema is missing or is not an object.
func schemaTypeOrUnknown(rawSchema json.RawMessage) string {
	decoded, ok := decodeSchema(rawSchema)
	if !ok {
		return "unknown"
	}

	return schemaToType(decoded)
}

func renderToolExamples(tools []ai.Tool) []string {
	if len(tools) == 1 {
		return []string{
			fmt.Sprintf("const %s = await %s;", defaultExampleVariable, renderToolExampleCall(tools[0])),
			fmt.Sprintf("return %s;", renderToolProjection(defaultExampleVariable, tools[0])),
		}
	}

	names := make([]string, len(tools))
	for index, tool := range tools {
		names[index] = tool.Name
	}

	variables := uniqueToolVariableNames(names)

	lines := []string{
		fmt.Sprintf("const [%s] = await Promise.all([", strings.Join(variables, ", ")),
	}

	for _, tool := range tools {
		lines = append(lines, fmt.Sprintf("  %s,", renderToolExampleCall(tool)))
	}

	lines = append(lines, "]);", "return {")

	for index, tool := range tools {
		lines = append(lines, fmt.Sprintf("  %s: %s,", formatObjectKey(tool.Name), renderToolProjection(variables[index], tool)))
	}

	return append(lines, "};")
}

// renderToolProjection renders the value an example returns for a tool's
// result: the first object property of its output schema, or the whole
// result when the output has no object properties.
func renderToolProjection(variable string, tool ai.Tool) string {
	field, ok := firstObjectProperty(tool.OutputSchema)
	if !ok {
		return variable
	}

	return fmt.Sprintf("{ %s: %s%s }", formatObjectKey(field), variable, formatPropertyAccess(field))
}

// firstObjectProperty returns the name of the first property of an output
// schema whose schema is an object.
func firstObjectProperty(rawSchema json.RawMessage) (string, bool) {
	outputSchema, ok := decodeSchema(rawSchema)
	if !ok {
		return "", false
	}

	properties := objectProperties(outputSchema)
	if len(properties.keys) == 0 {
		return "", false
	}

	return properties.keys[0], true
}

// renderToolExampleCall renders a call with the tool's first input example,
// or with an input sampled from its schema.
func renderToolExampleCall(tool ai.Tool) string {
	return fmt.Sprintf("tools%s(%s)", formatPropertyAccess(tool.Name), exampleInput(tool))
}

func exampleInput(tool ai.Tool) string {
	if len(tool.InputExamples) > 0 {
		if example, ok := decodeSchemaValue(tool.InputExamples[0]); ok {
			return encodeSchemaValue(example)
		}
	}

	inputSchema, ok := decodeSchema(tool.InputSchema)
	if !ok {
		return "{}"
	}

	return encodeSchemaValue(sampleFromSchema(inputSchema))
}

func formatObjectKey(key string) string {
	if identifierPattern.MatchString(key) {
		return key
	}

	return encodeSchemaValue(key)
}

func formatPropertyAccess(key string) string {
	if identifierPattern.MatchString(key) {
		return "." + key
	}

	return "[" + encodeSchemaValue(key) + "]"
}

// uniqueToolVariableNames derives distinct JavaScript identifiers from tool
// names.
func uniqueToolVariableNames(toolNames []string) []string {
	seen := make(map[string]int)
	variables := make([]string, len(toolNames))

	for index, toolName := range toolNames {
		base := toIdentifier(toolName)
		count := seen[base]
		seen[base] = count + 1

		variables[index] = base
		if count > 0 {
			variables[index] = fmt.Sprintf("%s%d", base, count+1)
		}
	}

	return variables
}

func toIdentifier(value string) string {
	identifier := identifierInvalidPattern.ReplaceAllString(value, "_")
	identifier = identifierLeadingPattern.ReplaceAllString(identifier, "")

	if identifier == "" {
		return defaultIdentifierFallback
	}

	return identifier
}

func toComment(value string) string {
	escaped := strings.ReplaceAll(value, "*/", "* /")

	return strings.TrimSpace(commentWhitespacePattern.ReplaceAllString(escaped, " "))
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}

	return ""
}
