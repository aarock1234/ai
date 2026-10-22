package mcp

import (
	"context"
	"encoding/json"
	"fmt"
)

// Prompt is a prompt template listed by the server.
type Prompt struct {
	Name        string                     `json:"name"`
	Title       string                     `json:"title,omitempty"`
	Description string                     `json:"description,omitempty"`
	Arguments   []PromptArgument           `json:"arguments,omitempty"`
	Meta        map[string]json.RawMessage `json:"_meta,omitempty"`
}

// PromptArgument is an argument a prompt template accepts.
type PromptArgument struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Required    bool   `json:"required,omitempty"`
}

// ListPromptsResult is one page of the server's prompts.
type ListPromptsResult struct {
	Prompts []Prompt `json:"prompts"`

	// NextCursor requests the following page; it is empty on the last page.
	NextCursor string                     `json:"nextCursor,omitempty"`
	Meta       map[string]json.RawMessage `json:"_meta,omitempty"`
}

func (r *ListPromptsResult) validate(raw json.RawMessage) error {
	return requireFields(raw, "prompts")
}

// GetPromptParams are the parameters of prompts/get.
type GetPromptParams struct {
	Name      string                     `json:"name"`
	Arguments map[string]string          `json:"arguments,omitempty"`
	Meta      map[string]json.RawMessage `json:"_meta,omitempty"`
}

// PromptRole is the role of a prompt message.
type PromptRole string

// Prompt message roles.
const (
	PromptRoleUser      PromptRole = "user"
	PromptRoleAssistant PromptRole = "assistant"
)

// PromptMessage is one message of a rendered prompt. Its content is text, an
// image, an embedded resource or a resource link.
type PromptMessage struct {
	Role    PromptRole `json:"role"`
	Content Content    `json:"content"`
}

// GetPromptResult is a rendered prompt.
type GetPromptResult struct {
	Description string                     `json:"description,omitempty"`
	Messages    []PromptMessage            `json:"messages"`
	Meta        map[string]json.RawMessage `json:"_meta,omitempty"`
}

func (r *GetPromptResult) validate(raw json.RawMessage) error {
	if err := requireFields(raw, "messages"); err != nil {
		return err
	}

	for index, message := range r.Messages {
		if message.Role != PromptRoleUser && message.Role != PromptRoleAssistant {
			return fmt.Errorf("prompt message %d has invalid role %q", index, message.Role)
		}

		if !isPromptContentType(message.Content.Type) {
			return fmt.Errorf("prompt message %d has unsupported content type %q", index, message.Content.Type)
		}
	}

	return nil
}

func isPromptContentType(contentType ContentType) bool {
	switch contentType {
	case ContentTypeText, ContentTypeImage, ContentTypeResource, ContentTypeResourceLink:
		return true
	default:
		return false
	}
}

// ListPrompts returns one page of the server's prompts. Upstream marks prompt
// support experimental.
func (c *Client) ListPrompts(ctx context.Context, params PaginatedParams) (*ListPromptsResult, error) {
	var result ListPromptsResult

	err := c.request(ctx, rpcCall{
		method: "prompts/list",
		params: params,
		result: &result,
	})
	if err != nil {
		return nil, err
	}

	return &result, nil
}

// GetPrompt renders a prompt with arguments. Upstream marks prompt support
// experimental.
func (c *Client) GetPrompt(ctx context.Context, params GetPromptParams) (*GetPromptResult, error) {
	var result GetPromptResult

	err := c.request(ctx, rpcCall{
		method: "prompts/get",
		params: params,
		result: &result,
	})
	if err != nil {
		return nil, err
	}

	return &result, nil
}
