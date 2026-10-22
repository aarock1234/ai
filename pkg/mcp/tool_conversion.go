package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/aarock1234/ai/internal/schema"
	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// Keys of the ai.Tool.Metadata of tools created from MCP definitions.
const (
	// ToolMetadataClientName holds the client name (string).
	ToolMetadataClientName = "clientName"

	// ToolMetadataToolName holds the server's tool name (string).
	ToolMetadataToolName = "toolName"

	// ToolMetadataTitle holds the tool title (string), when known.
	ToolMetadataTitle = "title"

	// ToolMetadataAnnotations holds the server's ToolAnnotations, when
	// reported.
	ToolMetadataAnnotations = "annotations"

	// ToolMetadataApp holds the AppToolMeta of tools with an MCP App
	// resource.
	ToolMetadataApp = "app"

	// ToolMetadataMeta holds the tool definition's _meta
	// (map[string]json.RawMessage), when present.
	ToolMetadataMeta = "_meta"
)

// ToolsOptions configure Tools and ToolsFromDefinitions.
type ToolsOptions struct {
	// Schemas selects tools by name and declares their schemas, creating
	// function tools instead of dynamic ones. Tools the server lists but
	// Schemas omits are skipped. Nil converts every tool automatically.
	Schemas map[string]ToolSchema
}

// ToolSchema declares the schemas of a tool explicitly.
type ToolSchema struct {
	// InputSchema is the JSON Schema object declared to the model.
	InputSchema json.RawMessage

	// OutputSchema, when set, validates the tool's structured content, or
	// its first text block parsed as JSON, and makes that JSON the tool
	// output instead of the CallToolResult.
	OutputSchema json.RawMessage
}

// NewToolSchema returns a ToolSchema whose input schema is generated from
// Input.
func NewToolSchema[Input any]() (ToolSchema, error) {
	inputSchema, err := schema.Generate[Input]()
	if err != nil {
		return ToolSchema{}, fmt.Errorf("generate tool input schema: %w", err)
	}

	return ToolSchema{
		InputSchema: inputSchema,
	}, nil
}

// NewToolSchemaWithOutput returns a ToolSchema whose input and output
// schemas are generated from Input and Output.
func NewToolSchemaWithOutput[Input, Output any]() (ToolSchema, error) {
	toolSchema, err := NewToolSchema[Input]()
	if err != nil {
		return ToolSchema{}, err
	}

	outputSchema, err := schema.Generate[Output]()
	if err != nil {
		return ToolSchema{}, fmt.Errorf("generate tool output schema: %w", err)
	}

	toolSchema.OutputSchema = outputSchema

	return toolSchema, nil
}

// Tools lists every tool on the server, following pagination, and converts
// them with ToolsFromDefinitions.
func (c *Client) Tools(ctx context.Context, options ToolsOptions) ([]ai.Tool, error) {
	page, err := c.ListTools(ctx, PaginatedParams{})
	if err != nil {
		return nil, err
	}

	definitions := *page

	for page.NextCursor != "" {
		page, err = c.ListTools(ctx, PaginatedParams{
			Cursor: page.NextCursor,
		})
		if err != nil {
			return nil, err
		}

		definitions.Tools = append(definitions.Tools, page.Tools...)
	}

	definitions.NextCursor = ""

	return c.ToolsFromDefinitions(definitions, options)
}

// ToolsFromDefinitions converts tool definitions into ai.Tool values in the
// server's order, without contacting the server. Calls run tools/call on this
// client; the output is the *CallToolResult, which is converted into text and
// image content for the model. Metadata holds the ToolMetadata keys.
func (c *Client) ToolsFromDefinitions(definitions ListToolsResult, options ToolsOptions) ([]ai.Tool, error) {
	definitions = c.prepareToolDefinitions(definitions, false)

	tools := make([]ai.Tool, 0, len(definitions.Tools))
	for _, definition := range definitions.Tools {
		var declared *ToolSchema
		if options.Schemas != nil {
			toolSchema, ok := options.Schemas[definition.Name]
			if !ok {
				continue
			}

			declared = &toolSchema
		}

		tool, err := c.convertTool(definition, declared)
		if err != nil {
			return nil, err
		}

		tools = append(tools, tool)
	}

	return tools, nil
}

func (c *Client) convertTool(definition ToolDefinition, declared *ToolSchema) (ai.Tool, error) {
	metadata, err := c.toolMetadata(definition)
	if err != nil {
		return ai.Tool{}, fmt.Errorf("convert mcp tool %q: %w", definition.Name, err)
	}

	tool := ai.Tool{
		Name:          definition.Name,
		Description:   definition.Description,
		Title:         toolTitle(definition),
		Metadata:      metadata,
		ToModelOutput: toModelOutput,
	}

	if declared == nil {
		inputSchema, err := dynamicInputSchema(definition.InputSchema)
		if err != nil {
			return ai.Tool{}, fmt.Errorf("convert mcp tool %q: %w", definition.Name, err)
		}

		tool.InputSchema = inputSchema
		tool.Execute = c.toolExecutor(definition.Name, nil)

		return ai.DynamicTool(tool), nil
	}

	outputValidator, err := compileOutputSchema(declared.OutputSchema)
	if err != nil {
		return ai.Tool{}, fmt.Errorf("compile output schema of mcp tool %q: %w", definition.Name, err)
	}

	tool.Kind = ai.ToolKindFunction
	tool.InputSchema = declared.InputSchema
	tool.OutputSchema = declared.OutputSchema
	tool.Execute = c.toolExecutor(definition.Name, outputValidator)

	return tool, nil
}

func toolTitle(definition ToolDefinition) string {
	if definition.Title != "" || definition.Annotations == nil {
		return definition.Title
	}

	return definition.Annotations.Title
}

func (c *Client) toolMetadata(definition ToolDefinition) (map[string]any, error) {
	metadata := map[string]any{
		ToolMetadataClientName: c.clientInfo.Name,
		ToolMetadataToolName:   definition.Name,
	}

	if title := toolTitle(definition); title != "" {
		metadata[ToolMetadataTitle] = title
	}

	if definition.Annotations != nil {
		metadata[ToolMetadataAnnotations] = *definition.Annotations
	}

	appMeta, err := appToolMetaOf(definition.Meta)
	if err != nil {
		return nil, err
	}

	if appMeta != nil && appMeta.ResourceURI != "" {
		appMeta.MIMEType = AppMIMEType
		metadata[ToolMetadataApp] = *appMeta
	}

	if definition.Meta != nil {
		metadata[ToolMetadataMeta] = definition.Meta
	}

	return metadata, nil
}

// dynamicInputSchema closes the server's input schema to additional
// properties and gives it properties when it has none.
func dynamicInputSchema(inputSchema json.RawMessage) (json.RawMessage, error) {
	var members map[string]json.RawMessage
	if err := json.Unmarshal(inputSchema, &members); err != nil {
		return nil, fmt.Errorf("decode input schema: %w", err)
	}

	if members == nil {
		members = make(map[string]json.RawMessage)
	}

	if !hasMember(members, "properties") {
		members["properties"] = json.RawMessage(`{}`)
	}

	members["additionalProperties"] = json.RawMessage(`false`)

	return json.Marshal(members)
}

func compileOutputSchema(outputSchema json.RawMessage) (*schema.Validator, error) {
	if len(outputSchema) == 0 {
		return nil, nil
	}

	return schema.Compile(outputSchema)
}

// toolExecutor returns an Execute function calling the named tool. With an
// output validator, successful calls return the validated JSON output.
func (c *Client) toolExecutor(name string, outputValidator *schema.Validator) ai.ToolExecuteFunc {
	return func(ctx context.Context, input json.RawMessage, _ ai.ToolExecutionOptions) (any, error) {
		result, err := c.CallTool(ctx, CallToolParams{
			Name:      name,
			Arguments: input,
		})
		if err != nil {
			return nil, err
		}

		if result.IsError || outputValidator == nil {
			return result, nil
		}

		return extractStructuredContent(result, outputValidator, name)
	}
}

// extractStructuredContent returns the structured content, or the first
// text block parsed as JSON, after validating it against the output schema.
func extractStructuredContent(result *CallToolResult, validator *schema.Validator, name string) (json.RawMessage, error) {
	if len(result.StructuredContent) > 0 && string(result.StructuredContent) != "null" {
		if err := validator.Validate(result.StructuredContent); err != nil {
			return nil, &ClientError{
				Message: fmt.Sprintf("tool %q returned structuredContent that does not match the expected outputSchema", name),
				Cause:   err,
			}
		}

		return result.StructuredContent, nil
	}

	for _, content := range result.Content {
		if content.Type != ContentTypeText {
			continue
		}

		output := json.RawMessage(content.Text)
		if !json.Valid(output) {
			return nil, &ClientError{
				Message: fmt.Sprintf("tool %q returned content that does not match the expected outputSchema", name),
				Cause:   errors.New("text content is not valid JSON"),
			}
		}

		if err := validator.Validate(output); err != nil {
			return nil, &ClientError{
				Message: fmt.Sprintf("tool %q returned content that does not match the expected outputSchema", name),
				Cause:   err,
			}
		}

		return output, nil
	}

	return nil, &ClientError{
		Message: fmt.Sprintf("tool %q did not return structuredContent or parseable text content", name),
	}
}

// toModelOutput converts a tool output into the result sent to the model:
// the content of a CallToolResult as text and files, and any other output,
// such as validated structured content, as JSON.
func toModelOutput(_ context.Context, event ai.ToolOutputEvent[json.RawMessage, any]) (model.ToolResultOutput, error) {
	result, encoded, err := callToolResultOf(event.Output)
	if err != nil {
		return nil, err
	}

	if result == nil || result.Content == nil {
		return model.JSONOutput{
			Value: encoded,
		}, nil
	}

	parts := make([]model.ToolResultContent, 0, len(result.Content))
	for _, content := range result.Content {
		part, err := contentToModel(content)
		if err != nil {
			return nil, err
		}

		parts = append(parts, part)
	}

	return model.ContentOutput{
		Value: parts,
	}, nil
}

// callToolResultOf returns output as a CallToolResult when it is one,
// including outputs restored from serialized history, together with the
// JSON encoding of output.
func callToolResultOf(output any) (*CallToolResult, json.RawMessage, error) {
	encoded, err := json.Marshal(output)
	if err != nil {
		return nil, nil, fmt.Errorf("encode mcp tool output: %w", err)
	}

	if result, ok := output.(*CallToolResult); ok {
		return result, encoded, nil
	}

	var probe struct {
		Content json.RawMessage `json:"content"`
	}

	if err := json.Unmarshal(encoded, &probe); err != nil || !isJSONArray(probe.Content) {
		return nil, encoded, nil
	}

	var result CallToolResult
	if err := json.Unmarshal(encoded, &result); err != nil {
		return nil, encoded, nil
	}

	return &result, encoded, nil
}

func isJSONArray(raw json.RawMessage) bool {
	return len(raw) > 0 && raw[0] == '['
}

// contentToModel converts text and images; other blocks are sent as their
// JSON encoding.
func contentToModel(content Content) (model.ToolResultContent, error) {
	switch content.Type {
	case ContentTypeText:
		return model.ToolResultText{
			Text: content.Text,
		}, nil
	case ContentTypeImage:
		return model.ToolResultFile{
			MediaType: content.MIMEType,
			Data: model.FileBytes{
				Data: content.Data,
			},
		}, nil
	default:
		encoded, err := json.Marshal(content)
		if err != nil {
			return nil, fmt.Errorf("encode mcp content: %w", err)
		}

		return model.ToolResultText{
			Text: string(encoded),
		}, nil
	}
}
