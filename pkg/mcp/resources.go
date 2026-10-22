package mcp

import (
	"context"
	"encoding/json"
)

// Resource is a resource listed by the server.
type Resource struct {
	URI         string                     `json:"uri"`
	Name        string                     `json:"name"`
	Title       string                     `json:"title,omitempty"`
	Description string                     `json:"description,omitempty"`
	MIMEType    string                     `json:"mimeType,omitempty"`
	Size        *float64                   `json:"size,omitempty"`
	Meta        map[string]json.RawMessage `json:"_meta,omitempty"`
}

// ListResourcesResult is one page of the server's resources.
type ListResourcesResult struct {
	Resources []Resource `json:"resources"`

	// NextCursor requests the following page; it is empty on the last page.
	NextCursor string                     `json:"nextCursor,omitempty"`
	Meta       map[string]json.RawMessage `json:"_meta,omitempty"`
}

func (r *ListResourcesResult) validate(raw json.RawMessage) error {
	return requireFields(raw, "resources")
}

// ReadResourceParams are the parameters of resources/read.
type ReadResourceParams struct {
	URI  string                     `json:"uri"`
	Meta map[string]json.RawMessage `json:"_meta,omitempty"`
}

// ReadResourceResult holds the contents of a resource.
type ReadResourceResult struct {
	Contents []ResourceContents         `json:"contents"`
	Meta     map[string]json.RawMessage `json:"_meta,omitempty"`
}

func (r *ReadResourceResult) validate(raw json.RawMessage) error {
	return requireFields(raw, "contents")
}

// ResourceTemplate describes parameterized resources by an RFC 6570 URI
// template.
type ResourceTemplate struct {
	URITemplate string                     `json:"uriTemplate"`
	Name        string                     `json:"name"`
	Title       string                     `json:"title,omitempty"`
	Description string                     `json:"description,omitempty"`
	MIMEType    string                     `json:"mimeType,omitempty"`
	Meta        map[string]json.RawMessage `json:"_meta,omitempty"`
}

// ListResourceTemplatesResult holds the server's resource templates.
type ListResourceTemplatesResult struct {
	ResourceTemplates []ResourceTemplate         `json:"resourceTemplates"`
	Meta              map[string]json.RawMessage `json:"_meta,omitempty"`
}

func (r *ListResourceTemplatesResult) validate(raw json.RawMessage) error {
	return requireFields(raw, "resourceTemplates")
}

// ListResources returns one page of the server's resources.
func (c *Client) ListResources(ctx context.Context, params PaginatedParams) (*ListResourcesResult, error) {
	var result ListResourcesResult

	err := c.request(ctx, rpcCall{
		method: "resources/list",
		params: params,
		result: &result,
	})
	if err != nil {
		return nil, err
	}

	return &result, nil
}

// ReadResource reads the contents of a resource.
func (c *Client) ReadResource(ctx context.Context, params ReadResourceParams) (*ReadResourceResult, error) {
	var result ReadResourceResult

	err := c.request(ctx, rpcCall{
		method: "resources/read",
		params: params,
		result: &result,
	})
	if err != nil {
		return nil, err
	}

	return &result, nil
}

// ListResourceTemplates returns the server's resource templates.
func (c *Client) ListResourceTemplates(ctx context.Context) (*ListResourceTemplatesResult, error) {
	var result ListResourceTemplatesResult

	err := c.request(ctx, rpcCall{
		method: "resources/templates/list",
		result: &result,
	})
	if err != nil {
		return nil, err
	}

	return &result, nil
}
