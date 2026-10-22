package fal

import (
	"fmt"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// videoOptionKeys are the VideoOptions members, which are mapped to request
// fields rather than passed through.
var videoOptionKeys = map[string]bool{
	"loop":            true,
	"motionStrength":  true,
	"resolution":      true,
	"negativePrompt":  true,
	"promptOptimizer": true,
}

// VideoOptions are the provider options of video models, read from the
// "fal" namespace. Other members of the namespace are sent in the request
// body unchanged.
type VideoOptions struct {
	// Loop makes the video loop. Only Luma models support it.
	Loop *bool `json:"loop,omitempty"`

	// MotionStrength is the motion strength, from 0 to 1.
	MotionStrength *float64 `json:"motionStrength,omitempty"`

	// Resolution is a model-specific resolution, such as "720p".
	Resolution *string `json:"resolution,omitempty"`

	NegativePrompt  *string `json:"negativePrompt,omitempty"`
	PromptOptimizer *bool   `json:"promptOptimizer,omitempty"`
}

// readVideoOptions reads the typed options and the passthrough members of
// the "fal" namespace.
func readVideoOptions(options model.ProviderOptions) (VideoOptions, jsonobject.Object, error) {
	videoOptions, _, err := model.ProviderValue[VideoOptions](options, Namespace)
	if err != nil {
		return VideoOptions{}, nil, err
	}

	if strength := videoOptions.MotionStrength; strength != nil && (*strength < 0 || *strength > 1) {
		return VideoOptions{}, nil, fmt.Errorf("%w: fal motionStrength must be between 0 and 1", model.ErrInvalidArgument)
	}

	raw, _, err := model.ProviderValue[jsonobject.Object](options, Namespace)
	if err != nil {
		return VideoOptions{}, nil, err
	}

	var passthrough jsonobject.Object
	for _, member := range raw {
		if !videoOptionKeys[member.Key] {
			passthrough = append(passthrough, member)
		}
	}

	return videoOptions, passthrough, nil
}
