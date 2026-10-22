package groq

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// Browser search tool identifiers.
const (
	browserSearchID   = "groq.browser_search"
	browserSearchName = "browser_search"
	browserSearchType = "browser_search"
)

// modelsWithoutBrowserSearch are the known models without native browser
// search. Unknown models are assumed to support it, so that newly released
// models keep requested tools.
var modelsWithoutBrowserSearch = map[string]bool{
	"gemma2-9b-it":                                  true,
	"llama-3.1-8b-instant":                          true,
	"llama-3.3-70b-versatile":                       true,
	"meta-llama/llama-guard-4-12b":                  true,
	"deepseek-r1-distill-llama-70b":                 true,
	"meta-llama/llama-4-maverick-17b-128e-instruct": true,
	"meta-llama/llama-4-scout-17b-16e-instruct":     true,
	"meta-llama/llama-prompt-guard-2-22m":           true,
	"meta-llama/llama-prompt-guard-2-86m":           true,
	"moonshotai/kimi-k2-instruct-0905":              true,
	"qwen/qwen3.6-27b":                              true,
	"llama-guard-3-8b":                              true,
	"llama3-70b-8192":                               true,
	"llama3-8b-8192":                                true,
	"mixtral-8x7b-32768":                            true,
	"qwen-qwq-32b":                                  true,
	"qwen-2.5-32b":                                  true,
	"deepseek-r1-distill-qwen-32b":                  true,
}

// BrowserSearch returns Groq's browser search tool, which navigates websites
// interactively on Groq's servers. It takes no arguments and is activated by
// the prompt. Models such as "openai/gpt-oss-120b" support it.
func BrowserSearch() model.ProviderTool {
	return model.ProviderTool{
		ID:   browserSearchID,
		Name: browserSearchName,
		Args: json.RawMessage("{}"),
	}
}

// preparedTools are the wire tools, tool choice and warnings of a call.
type preparedTools struct {
	Tools      []tool
	ToolChoice *toolChoice
	Warnings   []model.Warning
}

// prepareTools converts the call's tools. Browser search is sent only to
// models that support it.
func prepareTools(tools []model.Tool, choice *model.ToolChoice, modelID string) (preparedTools, error) {
	var prepared preparedTools
	if len(tools) == 0 {
		return prepared, nil
	}

	prepared.Tools = []tool{}
	for _, definition := range tools {
		switch definition := definition.(type) {
		case model.FunctionTool:
			prepared.Tools = append(prepared.Tools, tool{
				Type: "function",
				Function: &toolFunction{
					Name:        definition.Name,
					Description: definition.Description,
					Parameters:  definition.InputSchema,
					Strict:      definition.Strict,
				},
			})

		case model.ProviderTool:
			converted, warning := providerTool(definition, modelID)
			if warning != nil {
				prepared.Warnings = append(prepared.Warnings, *warning)

				continue
			}

			prepared.Tools = append(prepared.Tools, converted)

		default:
			return prepared, fmt.Errorf("%w: tool type %T", model.ErrUnsupported, definition)
		}
	}

	if choice == nil {
		return prepared, nil
	}

	switch choice.Type {
	case model.ToolChoiceAuto, model.ToolChoiceNone, model.ToolChoiceRequired:
		prepared.ToolChoice = &toolChoice{
			Mode: string(choice.Type),
		}

	case model.ToolChoiceTool:
		prepared.ToolChoice = &toolChoice{
			FunctionName: choice.ToolName,
		}

	default:
		return prepared, fmt.Errorf("%w: tool choice type %q", model.ErrUnsupported, choice.Type)
	}

	return prepared, nil
}

// providerTool converts a provider tool, or returns a warning when the tool
// or the model does not support it.
func providerTool(definition model.ProviderTool, modelID string) (tool, *model.Warning) {
	feature := "provider-defined tool " + definition.ID

	if definition.ID != browserSearchID {
		warning := model.UnsupportedWarning(feature, "")

		return tool{}, &warning
	}

	if modelsWithoutBrowserSearch[modelID] {
		warning := model.UnsupportedWarning(feature, fmt.Sprintf("Browser search is not supported on model %s.", modelID))

		return tool{}, &warning
	}

	return tool{
		Type: browserSearchType,
	}, nil
}
