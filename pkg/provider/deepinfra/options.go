package deepinfra

import (
	"fmt"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// ImageResponseFormat is the format of generated images.
type ImageResponseFormat string

// ImageResponseFormatB64JSON returns images as base64 JSON strings.
const ImageResponseFormatB64JSON ImageResponseFormat = "b64_json"

// ImageOptions are the common provider options of image models, read from
// the "deepinfra" namespace. Every key of the namespace, including keys not
// listed here, is sent in the request unchanged, so the JSON names follow the
// wire format.
type ImageOptions struct {
	// NegativePrompt describes what the image should not contain.
	NegativePrompt *string `json:"negative_prompt,omitempty"`

	// NumInferenceSteps is the number of denoising steps.
	NumInferenceSteps *float64 `json:"num_inference_steps,omitempty"`

	// GuidanceScale and Guidance control how closely the image follows the
	// prompt. Models accept one or the other.
	GuidanceScale *float64 `json:"guidance_scale,omitempty"`
	Guidance      *float64 `json:"guidance,omitempty"`

	// ResponseFormat selects the format of generated images.
	ResponseFormat ImageResponseFormat `json:"response_format,omitempty"`

	// Quality and Style are model-specific image settings.
	Quality *string `json:"quality,omitempty"`
	Style   *string `json:"style,omitempty"`

	// User identifies the end user.
	User *string `json:"user,omitempty"`
}

// readImageOptions validates the call's DeepInfra image options and returns
// every member of the namespace, in order, for the request.
func readImageOptions(options model.ProviderOptions) (jsonobject.Object, error) {
	imageOptions, _, err := model.ProviderValue[ImageOptions](options, Namespace)
	if err != nil {
		return nil, err
	}

	switch imageOptions.ResponseFormat {
	case "", ImageResponseFormatB64JSON:
	default:
		return nil, fmt.Errorf("%w: deepinfra response_format %q", model.ErrInvalidArgument, imageOptions.ResponseFormat)
	}

	members, _, err := model.ProviderValue[jsonobject.Object](options, Namespace)
	if err != nil {
		return nil, err
	}

	return members, nil
}
