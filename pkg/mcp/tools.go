package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/http"
)

// ToolDefinition is a tool as listed by the server.
type ToolDefinition struct {
	Name        string `json:"name"`
	Title       string `json:"title,omitempty"`
	Description string `json:"description,omitempty"`

	// InputSchema is the JSON Schema object of the tool's arguments.
	InputSchema json.RawMessage `json:"inputSchema"`

	// OutputSchema is the JSON Schema of the tool's structured content.
	OutputSchema json.RawMessage `json:"outputSchema,omitempty"`

	Annotations *ToolAnnotations           `json:"annotations,omitempty"`
	Meta        map[string]json.RawMessage `json:"_meta,omitempty"`
}

// ToolAnnotations are behavioral hints reported by the server for a tool.
// They are untrusted unless the server itself is trusted.
type ToolAnnotations struct {
	Title           string `json:"title,omitempty"`
	ReadOnlyHint    *bool  `json:"readOnlyHint,omitempty"`
	DestructiveHint *bool  `json:"destructiveHint,omitempty"`
	IdempotentHint  *bool  `json:"idempotentHint,omitempty"`
	OpenWorldHint   *bool  `json:"openWorldHint,omitempty"`
}

// ListToolsResult is one page of the server's tools.
type ListToolsResult struct {
	Tools []ToolDefinition `json:"tools"`

	// NextCursor requests the following page; it is empty on the last page.
	NextCursor string                     `json:"nextCursor,omitempty"`
	Meta       map[string]json.RawMessage `json:"_meta,omitempty"`
}

func (r *ListToolsResult) validate(raw json.RawMessage) error {
	if err := requireFields(raw, "tools"); err != nil {
		return err
	}

	for _, tool := range r.Tools {
		if !isJSONObject(tool.InputSchema) {
			return fmt.Errorf("tool %q must have an inputSchema object", tool.Name)
		}

		if tool.OutputSchema != nil && !isJSONObject(tool.OutputSchema) {
			return fmt.Errorf("tool %q has an outputSchema that is not an object", tool.Name)
		}
	}

	return nil
}

// CallToolParams are the parameters of tools/call.
type CallToolParams struct {
	Name string `json:"name"`

	// Arguments is the JSON object of tool arguments. Empty sends {}.
	Arguments json.RawMessage `json:"arguments"`

	Meta map[string]json.RawMessage `json:"_meta,omitempty"`
}

// CallToolResult is the result of a tool call.
type CallToolResult struct {
	// Content is the unstructured result. Results with only structured
	// content get a text block with its JSON encoding; results in the legacy
	// toolResult format have none.
	Content []Content `json:"content"`

	// StructuredContent is the result as JSON, validated by the server
	// against the tool's output schema.
	StructuredContent json.RawMessage `json:"structuredContent,omitempty"`

	// IsError reports a tool failure. The content describes it to the model.
	IsError bool `json:"isError,omitempty"`

	// ToolResult is the result in the 2024-10-07 protocol format.
	ToolResult json.RawMessage `json:"toolResult,omitempty"`

	ResultType string                     `json:"resultType,omitempty"`
	Meta       map[string]json.RawMessage `json:"_meta,omitempty"`
}

// validate accepts content, structured content or a legacy toolResult, and
// derives text content from structured content when there is no content.
func (r *CallToolResult) validate(raw json.RawMessage) error {
	var members map[string]json.RawMessage
	if err := json.Unmarshal(raw, &members); err != nil {
		return err
	}

	if hasMember(members, "content") {
		return nil
	}

	if structured, ok := members["structuredContent"]; ok {
		text, err := compactJSON(structured)
		if err != nil {
			return err
		}

		r.Content = []Content{
			{
				Type: ContentTypeText,
				Text: text,
			},
		}

		return nil
	}

	if _, ok := members["toolResult"]; ok {
		return nil
	}

	return errors.New("tool result must have content, structuredContent or toolResult")
}

func hasMember(members map[string]json.RawMessage, name string) bool {
	value, ok := members[name]

	return ok && string(value) != "null"
}

func compactJSON(raw json.RawMessage) (string, error) {
	var compacted bytes.Buffer
	if err := json.Compact(&compacted, raw); err != nil {
		return "", err
	}

	return compacted.String(), nil
}

// ListTools returns one page of the server's tools.
func (c *Client) ListTools(ctx context.Context, params PaginatedParams) (*ListToolsResult, error) {
	var result ListToolsResult

	err := c.request(ctx, rpcCall{
		method: "tools/list",
		params: params,
		result: &result,
	})
	if err != nil {
		return nil, err
	}

	prepared := c.prepareToolDefinitions(result, params.Cursor == "")

	return &prepared, nil
}

// CallTool calls a tool on the server. Transient failures are retried up to
// ClientOptions.MaxRetries times.
func (c *Client) CallTool(ctx context.Context, params CallToolParams) (*CallToolResult, error) {
	if len(bytes.TrimSpace(params.Arguments)) == 0 {
		params.Arguments = json.RawMessage(`{}`)
	}

	headers, err := c.toolRequestHeaders(params)
	if err != nil {
		return nil, err
	}

	call := func(ctx context.Context) (*CallToolResult, error) {
		var result CallToolResult

		err := c.request(ctx, rpcCall{
			method:  "tools/call",
			params:  params,
			headers: headers,
			result:  &result,
		})
		if err != nil {
			return nil, err
		}

		return &result, nil
	}

	return retryToolCall(ctx, c.maxRetries, call)
}

// prepareToolDefinitions records the x-mcp-header bindings of each tool for
// transports that send them, dropping tools with invalid annotations. A
// first page resets the bindings.
func (c *Client) prepareToolDefinitions(definitions ListToolsResult, reset bool) ListToolsResult {
	if !c.isModernProtocol() || !supportsToolParameterHeaders(c.transport) {
		return definitions
	}

	tools := make([]ToolDefinition, 0, len(definitions.Tools))
	bindingsByTool := make(map[string][]toolHeaderBinding, len(definitions.Tools))

	for _, tool := range definitions.Tools {
		bindings, err := toolHeaderBindings(tool.InputSchema)
		if err != nil {
			c.reportError(&ClientError{
				Message: fmt.Sprintf("ignoring mcp tool %q", tool.Name),
				Cause:   err,
			})

			continue
		}

		bindingsByTool[tool.Name] = bindings
		tools = append(tools, tool)
	}

	c.mu.Lock()
	if reset {
		clear(c.toolHeaderBindings)
	}
	maps.Copy(c.toolHeaderBindings, bindingsByTool)
	c.mu.Unlock()

	definitions.Tools = tools

	return definitions
}

func supportsToolParameterHeaders(transport Transport) bool {
	headerTransport, ok := transport.(ToolParameterHeaderTransport)

	return ok && headerTransport.SupportsToolParameterHeaders()
}

// toolRequestHeaders returns the headers mirrored from the call's arguments
// in the stateless protocol.
func (c *Client) toolRequestHeaders(params CallToolParams) (http.Header, error) {
	if !c.isModernProtocol() {
		return nil, nil
	}

	c.mu.Lock()
	bindings := c.toolHeaderBindings[params.Name]
	c.mu.Unlock()

	if len(bindings) == 0 {
		return nil, nil
	}

	arguments, ok := decodeArgumentsObject(params.Arguments)
	if !ok {
		return nil, nil
	}

	headers, err := createToolHeaders(bindings, arguments)
	if err != nil {
		return nil, &ClientError{
			Message: fmt.Sprintf("failed to create mcp headers for tool %q", params.Name),
			Cause:   err,
		}
	}

	return headers, nil
}

// decodeArgumentsObject decodes tool arguments, keeping numbers exact. It
// reports false when the arguments are not a JSON object.
func decodeArgumentsObject(raw json.RawMessage) (map[string]any, bool) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()

	var arguments map[string]any
	if err := decoder.Decode(&arguments); err != nil || arguments == nil {
		return nil, false
	}

	return arguments, true
}
