package togetherai

import (
	"encoding/json"

	"github.com/aarock1234/ai/pkg/model"
)

// ImageOptions are the provider options of image models, read from the
// "togetherai" namespace. Every key of the namespace, including keys not
// listed here, is sent in the request body unchanged, so the JSON names
// follow the wire format.
type ImageOptions struct {
	// Steps is the number of generation steps. Higher values can improve
	// quality.
	Steps *float64 `json:"steps,omitempty"`

	// Guidance is the guidance scale.
	Guidance *float64 `json:"guidance,omitempty"`

	// NegativePrompt describes what to avoid.
	NegativePrompt *string `json:"negative_prompt,omitempty"`

	// DisableSafetyChecker stops the API from rejecting images flagged as
	// potentially NSFW. It is not available for the Flux Schnell Free and
	// Flux Pro models.
	DisableSafetyChecker *bool `json:"disable_safety_checker,omitempty"`
}

// diffusionOptionKeys are the ImageOptions keys, which only diffusion models
// accept.
var diffusionOptionKeys = []string{
	"steps",
	"guidance",
	"negative_prompt",
	"disable_safety_checker",
}

// RerankingOptions are the provider options of reranking models, read from
// the "togetherai" namespace.
type RerankingOptions struct {
	// RankFields are the keys of JSON object documents to rank by. All keys
	// are used by default.
	RankFields []string `json:"rankFields,omitempty"`
}

// imageOptionMembers validates the image options and returns every member
// of the namespace, which is forwarded to the API.
func imageOptionMembers(options model.ProviderOptions) (map[string]json.RawMessage, error) {
	if _, _, err := model.ProviderValue[ImageOptions](options, Namespace); err != nil {
		return nil, err
	}

	members, _, err := model.ProviderValue[map[string]json.RawMessage](options, Namespace)
	if err != nil {
		return nil, err
	}

	return members, nil
}
