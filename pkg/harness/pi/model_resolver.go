package pi

import (
	"strings"
)

// DefaultGatewayModel is the model of turns that select none when the AI
// Gateway authenticates Pi. It is looked up under Pi's vercel-ai-gateway
// provider.
const DefaultGatewayModel = "anthropic/claude-sonnet-4.6"

// resolveModel ports upstream createPiModelResolver over the models Pi
// reports as available, which are the models whose provider is
// authenticated. With gateway credentials the gateway-routed entry wins:
// Pi lists the same ID under several providers. A "<provider>/<id>"
// reference resolves under that provider before an entry whose flat ID
// equals the whole reference, such as a routing proxy's entry.
func resolveModel(models []modelInfo, modelID string, usesGateway bool) (modelInfo, bool) {
	matches := func(model modelInfo) bool {
		return model.ID == modelID || model.Name == modelID
	}

	if usesGateway {
		for _, model := range models {
			if model.Provider == gatewayProvider && matches(model) {
				return model, true
			}
		}
	}

	scoped, hasScoped := scopedMatch(models, modelID)
	if hasScoped {
		return scoped, true
	}

	var flat []modelInfo
	for _, model := range models {
		if matches(model) {
			flat = append(flat, model)
		}
	}

	if len(flat) == 0 {
		return modelInfo{}, false
	}

	return flat[0], true
}

// scopedMatch returns the model a "<provider>/<id>" reference names.
func scopedMatch(models []modelInfo, modelID string) (modelInfo, bool) {
	provider, bareID, found := strings.Cut(modelID, "/")
	if !found {
		return modelInfo{}, false
	}

	for _, model := range models {
		if model.Provider == provider && (model.ID == bareID || model.Name == bareID) {
			return model, true
		}
	}

	return modelInfo{}, false
}
