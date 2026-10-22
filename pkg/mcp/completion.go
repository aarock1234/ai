package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
)

// maxCompletionValues is the most completion values a result may hold.
const maxCompletionValues = 100

// CompletionReferenceType selects what an argument completion refers to.
type CompletionReferenceType string

// Completion reference types.
const (
	CompletionReferencePrompt   CompletionReferenceType = "ref/prompt"
	CompletionReferenceResource CompletionReferenceType = "ref/resource"
)

// CompletionReference names the prompt (by Name) or the resource template
// (by URI) whose argument is completed.
type CompletionReference struct {
	Type CompletionReferenceType `json:"type"`
	Name string                  `json:"name,omitempty"`
	URI  string                  `json:"uri,omitempty"`
}

// CompletionArgument is the argument being completed and its partial value.
type CompletionArgument struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

// CompletionContext holds the values of arguments already provided.
type CompletionContext struct {
	Arguments map[string]string `json:"arguments"`
}

// CompleteParams are the parameters of completion/complete.
type CompleteParams struct {
	Ref      CompletionReference        `json:"ref"`
	Argument CompletionArgument         `json:"argument"`
	Context  *CompletionContext         `json:"context,omitempty"`
	Meta     map[string]json.RawMessage `json:"_meta,omitempty"`
}

// Completion holds suggested argument values.
type Completion struct {
	Values []string `json:"values"`

	// Total is the number of available values, which may exceed Values.
	Total *int `json:"total,omitempty"`

	// HasMore reports values beyond those returned.
	HasMore *bool `json:"hasMore,omitempty"`
}

// CompleteResult is the result of completion/complete.
type CompleteResult struct {
	Completion Completion                 `json:"completion"`
	Meta       map[string]json.RawMessage `json:"_meta,omitempty"`
}

func (r *CompleteResult) validate(raw json.RawMessage) error {
	if err := requireFields(raw, "completion"); err != nil {
		return err
	}

	if r.Completion.Values == nil {
		return errors.New("completion must have values")
	}

	if len(r.Completion.Values) > maxCompletionValues {
		return fmt.Errorf("completion has %d values, more than %d", len(r.Completion.Values), maxCompletionValues)
	}

	return nil
}

// Complete requests completion suggestions for a prompt or resource template
// argument.
func (c *Client) Complete(ctx context.Context, params CompleteParams) (*CompleteResult, error) {
	var result CompleteResult

	err := c.request(ctx, rpcCall{
		method: "completion/complete",
		params: params,
		result: &result,
	})
	if err != nil {
		return nil, err
	}

	return &result, nil
}
