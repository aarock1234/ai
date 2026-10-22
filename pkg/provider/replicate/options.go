package replicate

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// OutputFormat is the file format of generated images.
type OutputFormat string

// Output formats.
const (
	OutputPNG  OutputFormat = "png"
	OutputJPG  OutputFormat = "jpg"
	OutputWebP OutputFormat = "webp"
)

// ImageOptions are the provider options of image models, read from the
// "replicate" namespace. Models accept different inputs: every member other
// than the polling settings is sent in the prediction input, including
// model-specific members that are not fields here.
type ImageOptions struct {
	// MaxWaitTimeInSeconds is how long Replicate holds the creating request
	// open for the prediction to finish. Nil uses Replicate's default of 60
	// seconds.
	MaxWaitTimeInSeconds *float64 `json:"maxWaitTimeInSeconds,omitempty"`

	// PollIntervalMillis is the delay between status checks when the
	// prediction is still running after the synchronous wait. The default is
	// 500.
	PollIntervalMillis *int `json:"pollIntervalMillis,omitempty"`

	// MaxPollAttempts limits the status checks. The default is 240.
	MaxPollAttempts *int `json:"maxPollAttempts,omitempty"`

	// GuidanceScale is the classifier-free guidance scale.
	GuidanceScale *float64 `json:"guidance_scale,omitempty"`

	// NumInferenceSteps is the number of denoising steps.
	NumInferenceSteps *float64 `json:"num_inference_steps,omitempty"`

	// NegativePrompt describes what to avoid.
	NegativePrompt *string `json:"negative_prompt,omitempty"`

	// OutputFormat is the image file format.
	OutputFormat *OutputFormat `json:"output_format,omitempty"`

	// OutputQuality is the JPG and WebP quality, from 1 to 100.
	OutputQuality *float64 `json:"output_quality,omitempty"`

	// Strength is the image-to-image transformation strength, from 0 to 1.
	Strength *float64 `json:"strength,omitempty"`
}

// pollOptionKeys are the members that configure polling and are not sent
// in the prediction input.
var pollOptionKeys = []string{
	"maxWaitTimeInSeconds",
	"pollIntervalMillis",
	"maxPollAttempts",
}

// readImageOptions reads and validates the call's Replicate options and
// returns the members to send in the prediction input.
func readImageOptions(options model.ProviderOptions) (ImageOptions, map[string]json.RawMessage, error) {
	imageOptions, _, err := model.ProviderValue[ImageOptions](options, Namespace)
	if err != nil {
		return ImageOptions{}, nil, err
	}

	if err := imageOptions.validate(); err != nil {
		return ImageOptions{}, nil, err
	}

	input, _, err := model.ProviderValue[map[string]json.RawMessage](options, Namespace)
	if err != nil {
		return ImageOptions{}, nil, err
	}

	for _, key := range pollOptionKeys {
		delete(input, key)
	}

	return imageOptions, input, nil
}

func (o ImageOptions) validate() error {
	if o.MaxWaitTimeInSeconds != nil && *o.MaxWaitTimeInSeconds <= 0 {
		return fmt.Errorf("%w: replicate maxWaitTimeInSeconds must be positive", model.ErrInvalidArgument)
	}

	if o.PollIntervalMillis != nil && *o.PollIntervalMillis <= 0 {
		return fmt.Errorf("%w: replicate pollIntervalMillis must be positive", model.ErrInvalidArgument)
	}

	if o.MaxPollAttempts != nil && *o.MaxPollAttempts <= 0 {
		return fmt.Errorf("%w: replicate maxPollAttempts must be positive", model.ErrInvalidArgument)
	}

	if o.OutputFormat != nil {
		switch *o.OutputFormat {
		case OutputPNG, OutputJPG, OutputWebP:
		default:
			return fmt.Errorf("%w: replicate output_format %q", model.ErrInvalidArgument, *o.OutputFormat)
		}
	}

	if o.OutputQuality != nil && (*o.OutputQuality < 1 || *o.OutputQuality > 100) {
		return fmt.Errorf("%w: replicate output_quality must be between 1 and 100", model.ErrInvalidArgument)
	}

	if o.Strength != nil && (*o.Strength < 0 || *o.Strength > 1) {
		return fmt.Errorf("%w: replicate strength must be between 0 and 1", model.ErrInvalidArgument)
	}

	return nil
}
