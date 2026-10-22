package policyopa

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/ai/middleware"
	"github.com/aarock1234/ai/pkg/model"
)

// OPACapabilityMiddlewareOptions configure OPACapabilityMiddleware.
type OPACapabilityMiddlewareOptions struct {
	// Client evaluates the policy. It is required.
	Client PolicyClient

	// Path is the Rego entrypoint that returns the allowed tool names, such
	// as "agent/tools/allowed".
	Path string

	// ToInput shapes the OPA input of a model call. Nil, or a nil result,
	// uses the DefaultOPACapabilityInput.
	ToInput func(ctx context.Context, input DefaultOPACapabilityInput) any
}

// DefaultOPACapabilityInput is the OPA input of a model call when ToInput
// is not set. Messages are encoded in the upstream model message format.
type DefaultOPACapabilityInput struct {
	Messages        []model.Message
	ProviderOptions model.ProviderOptions
}

// defaultOPACapabilityInputJSON is the JSON encoding of
// DefaultOPACapabilityInput.
type defaultOPACapabilityInputJSON struct {
	Messages        []wirejson.Message    `json:"messages"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

// MarshalJSON encodes the input with its messages in the upstream model
// message format.
func (i DefaultOPACapabilityInput) MarshalJSON() ([]byte, error) {
	return json.Marshal(defaultOPACapabilityInputJSON{
		Messages:        policyJSON.Messages(i.Messages),
		ProviderOptions: i.ProviderOptions,
	})
}

// allowedToolsResult is the object form of an allowlist.
type allowedToolsResult struct {
	Tools json.RawMessage `json:"tools"`
}

// OPACapabilityMiddleware returns experimental language model middleware
// that keeps only the tools an OPA policy allows on every model call. Use
// it with the tool approval policy for defense in depth: the model does not
// see the tools it cannot call.
//
// The policy at Path returns a list of allowed tool names, or an object
// whose "tools" member is that list. Function tools are kept when their
// name is allowed, and provider tools when their ID or name is allowed.
// When the evaluation fails or the result is malformed, every tool is
// removed, so that a misconfiguration never widens capabilities.
func OPACapabilityMiddleware(options OPACapabilityMiddlewareOptions) middleware.Language {
	return middleware.Language{
		TransformParams: func(ctx context.Context, transform middleware.LanguageTransform) (model.CallOptions, error) {
			return restrictTools(ctx, options, transform.Params)
		},
	}
}

// restrictTools keeps the tools of params that the policy allows, or none
// when the policy cannot be evaluated or its result is malformed.
func restrictTools(ctx context.Context, options OPACapabilityMiddlewareOptions, params model.CallOptions) (model.CallOptions, error) {
	if len(params.Tools) == 0 {
		return params, nil
	}

	input := capabilityInput(ctx, options.ToInput, params)

	result, err := options.Client.Evaluate(ctx, options.Path, input)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return model.CallOptions{}, fmt.Errorf("evaluate tool capability policy: %w", ctxErr)
		}

		params.Tools = nil

		return params, nil
	}

	allowed, ok := allowedToolNames(result)
	if !ok {
		params.Tools = nil

		return params, nil
	}

	params.Tools = allowedTools(params.Tools, allowed)

	return params, nil
}

func capabilityInput(
	ctx context.Context,
	toInput func(ctx context.Context, input DefaultOPACapabilityInput) any,
	params model.CallOptions,
) any {
	input := DefaultOPACapabilityInput{
		Messages:        params.Prompt,
		ProviderOptions: params.ProviderOptions,
	}

	if toInput != nil {
		if custom := toInput(ctx, input); custom != nil {
			return custom
		}
	}

	return input
}

// allowedToolNames reads an allowlist: a list of strings, or an object with
// such a list in "tools". It reports false for any other result.
func allowedToolNames(result json.RawMessage) (map[string]bool, bool) {
	list := result

	var object allowedToolsResult
	if err := json.Unmarshal(result, &object); err == nil {
		list = object.Tools
	}

	names, ok := decodeJSONValue[[]string](list)
	if !ok {
		return nil, false
	}

	allowed := make(map[string]bool, len(names))
	for _, name := range names {
		allowed[name] = true
	}

	return allowed, true
}

// allowedTools returns the tools whose name, or provider tool ID, is
// allowed, in their original order.
func allowedTools(tools []model.Tool, allowed map[string]bool) []model.Tool {
	kept := make([]model.Tool, 0, len(tools))

	for _, tool := range tools {
		if isAllowedTool(tool, allowed) {
			kept = append(kept, tool)
		}
	}

	return kept
}

func isAllowedTool(tool model.Tool, allowed map[string]bool) bool {
	if providerTool, ok := tool.(model.ProviderTool); ok && allowed[providerTool.ID] {
		return true
	}

	return allowed[tool.ToolName()]
}
