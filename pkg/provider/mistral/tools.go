package mistral

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/toolnames"
	"github.com/aarock1234/ai/pkg/model"
)

// Provider tool IDs and their names on the wire.
const (
	webSearchID        = "mistral.web_search"
	webSearchPremiumID = "mistral.web_search_premium"

	webSearchName        = "web_search"
	webSearchPremiumName = "web_search_premium"
)

// providerToolNames maps provider tool IDs to Mistral's tool types.
var providerToolNames = map[string]string{
	webSearchID:        webSearchName,
	webSearchPremiumID: webSearchPremiumName,
}

// WebSearch returns Mistral's web search tool. It runs on Mistral's servers
// and is available only to conversation models.
func WebSearch() model.ProviderTool {
	return model.ProviderTool{
		ID:   webSearchID,
		Name: webSearchName,
		Args: json.RawMessage("{}"),
	}
}

// WebSearchPremium returns Mistral's web search tool with verified news
// sources. It is available only to conversation models.
func WebSearchPremium() model.ProviderTool {
	return model.ProviderTool{
		ID:   webSearchPremiumID,
		Name: webSearchPremiumName,
		Args: json.RawMessage("{}"),
	}
}

// toolChoice is a Mistral tool choice mode.
type toolChoice string

const (
	toolChoiceAuto toolChoice = "auto"
	toolChoiceNone toolChoice = "none"
	toolChoiceAny  toolChoice = "any"
)

// tool is a function tool or, in conversations, a built-in tool type.
type tool struct {
	Type     string        `json:"type"`
	Function *toolFunction `json:"function,omitempty"`
}

type toolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
	Strict      *bool           `json:"strict,omitempty"`
}

// preparedTools are the wire tools, tool choice and warnings of a call.
type preparedTools struct {
	Tools      []tool
	ToolChoice *toolChoice
	Warnings   []model.Warning
}

// toolSettings selects which provider tools a model accepts.
type toolSettings struct {
	// builtIn enables the web search tools of the Conversations API.
	builtIn bool
}

// prepareTools converts the call's tools. Mistral cannot force a specific
// tool, so a tool choice keeps only that tool and requires a call.
func prepareTools(tools []model.Tool, choice *model.ToolChoice, settings toolSettings) (preparedTools, error) {
	var prepared preparedTools
	if len(tools) == 0 {
		return prepared, nil
	}

	prepared.Tools = []tool{}
	for _, definition := range tools {
		converted, warning, err := convertTool(definition, settings)
		if err != nil {
			return prepared, err
		}

		if warning != nil {
			prepared.Warnings = append(prepared.Warnings, *warning)

			continue
		}

		prepared.Tools = append(prepared.Tools, converted)
	}

	if choice == nil {
		return prepared, nil
	}

	switch choice.Type {
	case model.ToolChoiceAuto:
		prepared.ToolChoice = new(toolChoiceAuto)

	case model.ToolChoiceNone:
		prepared.ToolChoice = new(toolChoiceNone)

	case model.ToolChoiceRequired:
		prepared.ToolChoice = new(toolChoiceAny)

	case model.ToolChoiceTool:
		mapping := toolnames.New(tools, providerToolNames)
		prepared.Tools = onlyTool(prepared.Tools, mapping.ToProvider(choice.ToolName))
		prepared.ToolChoice = new(toolChoiceAny)

	default:
		return prepared, fmt.Errorf("%w: tool choice type %q", model.ErrUnsupported, choice.Type)
	}

	return prepared, nil
}

// convertTool converts one tool, or returns a warning for a provider tool the
// model does not support.
func convertTool(definition model.Tool, settings toolSettings) (tool, *model.Warning, error) {
	switch definition := definition.(type) {
	case model.FunctionTool:
		return tool{
			Type: "function",
			Function: &toolFunction{
				Name:        definition.Name,
				Description: definition.Description,
				Parameters:  definition.InputSchema,
				Strict:      definition.Strict,
			},
		}, nil, nil

	case model.ProviderTool:
		toolType, known := providerToolNames[definition.ID]
		if settings.builtIn && known {
			return tool{
				Type: toolType,
			}, nil, nil
		}

		warning := model.UnsupportedWarning("provider-defined tool "+definition.ID, "")

		return tool{}, &warning, nil

	default:
		return tool{}, nil, fmt.Errorf("%w: tool type %T", model.ErrUnsupported, definition)
	}
}

// onlyTool keeps the function or built-in tool with the given wire name.
func onlyTool(tools []tool, name string) []tool {
	kept := []tool{}
	for _, candidate := range tools {
		if candidate.wireName() == name {
			kept = append(kept, candidate)
		}
	}

	return kept
}

func (t tool) wireName() string {
	if t.Function != nil {
		return t.Function.Name
	}

	return t.Type
}
