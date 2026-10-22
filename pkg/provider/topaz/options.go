package topaz

import (
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// EnhancementStrength is how aggressively an image is enhanced.
type EnhancementStrength string

// Enhancement strengths.
const (
	EnhancementLow    EnhancementStrength = "low"
	EnhancementMedium EnhancementStrength = "medium"
	EnhancementHigh   EnhancementStrength = "high"
)

// GrainModel is the film grain model.
type GrainModel string

// Grain models.
const (
	GrainSilver   GrainModel = "silver"
	GrainGaussian GrainModel = "gaussian"
	GrainGrey     GrainModel = "grey"
)

// OutputFormat is the file format of enhanced images.
type OutputFormat string

// Output formats.
const (
	OutputJPEG OutputFormat = "jpeg"
	OutputJPG  OutputFormat = "jpg"
	OutputPNG  OutputFormat = "png"
	OutputTIFF OutputFormat = "tiff"
	OutputTIF  OutputFormat = "tif"
)

// ImageOptions are the provider options of image models, read from the
// "topaz" namespace.
type ImageOptions struct {
	// EnhancementStrength is how aggressively the model enhances the image.
	// The default is high.
	EnhancementStrength *EnhancementStrength `json:"enhancementStrength,omitempty"`

	// Grain adds film grain. The default is false.
	Grain *bool `json:"grain,omitempty"`

	// GrainDensity is the grain intensity, from 0 to 1.
	GrainDensity *float64 `json:"grainDensity,omitempty"`

	// GrainModel is the grain model. The default is silver.
	GrainModel *GrainModel `json:"grainModel,omitempty"`

	// GrainSize is the grain particle size, from 1 to 5.
	GrainSize *float64 `json:"grainSize,omitempty"`

	// GrainStrength is the grain effect strength, from 0 to 1.
	GrainStrength *float64 `json:"grainStrength,omitempty"`

	// InputWidth and InputHeight are the input dimensions in pixels. Topaz
	// infers them from the upload when omitted.
	InputWidth  *int `json:"inputWidth,omitempty"`
	InputHeight *int `json:"inputHeight,omitempty"`

	// OutputWidth and OutputHeight are the output dimensions, from 1 to
	// 32000 pixels. They take precedence over the size call option.
	OutputWidth  *int `json:"outputWidth,omitempty"`
	OutputHeight *int `json:"outputHeight,omitempty"`

	// OutputFormat is the image file format.
	OutputFormat *OutputFormat `json:"outputFormat,omitempty"`

	// CropToFill crops the output to fill the requested dimensions.
	CropToFill *bool `json:"cropToFill,omitempty"`

	// WebhookURL receives job status webhooks.
	WebhookURL *string `json:"webhookUrl,omitempty"`

	// PollIntervalMillis is the delay between status checks. The default is
	// 2000.
	PollIntervalMillis *int `json:"pollIntervalMillis,omitempty"`

	// PollTimeoutMillis is how long to wait for the job. The default is
	// 600000 (10 minutes).
	PollTimeoutMillis *int `json:"pollTimeoutMillis,omitempty"`
}

func readImageOptions(options model.ProviderOptions) (ImageOptions, error) {
	imageOptions, _, err := model.ProviderValue[ImageOptions](options, Namespace)
	if err != nil {
		return ImageOptions{}, err
	}

	if err := imageOptions.validate(); err != nil {
		return ImageOptions{}, err
	}

	return imageOptions, nil
}

func (o ImageOptions) validate() error {
	if o.EnhancementStrength != nil {
		switch *o.EnhancementStrength {
		case EnhancementLow, EnhancementMedium, EnhancementHigh:
		default:
			return fmt.Errorf("%w: topaz enhancementStrength %q", model.ErrInvalidArgument, *o.EnhancementStrength)
		}
	}

	if o.GrainModel != nil {
		switch *o.GrainModel {
		case GrainSilver, GrainGaussian, GrainGrey:
		default:
			return fmt.Errorf("%w: topaz grainModel %q", model.ErrInvalidArgument, *o.GrainModel)
		}
	}

	if o.OutputFormat != nil {
		switch *o.OutputFormat {
		case OutputJPEG, OutputJPG, OutputPNG, OutputTIFF, OutputTIF:
		default:
			return fmt.Errorf("%w: topaz outputFormat %q", model.ErrInvalidArgument, *o.OutputFormat)
		}
	}

	ranges := []struct {
		name    string
		value   *float64
		minimum float64
		maximum float64
	}{
		{
			name:    "grainDensity",
			value:   o.GrainDensity,
			minimum: 0,
			maximum: 1,
		},
		{
			name:    "grainSize",
			value:   o.GrainSize,
			minimum: 1,
			maximum: 5,
		},
		{
			name:    "grainStrength",
			value:   o.GrainStrength,
			minimum: 0,
			maximum: 1,
		},
	}

	for _, check := range ranges {
		if check.value != nil && (*check.value < check.minimum || *check.value > check.maximum) {
			return fmt.Errorf("%w: topaz %s must be between %v and %v", model.ErrInvalidArgument, check.name, check.minimum, check.maximum)
		}
	}

	return o.validateIntegers()
}

func (o ImageOptions) validateIntegers() error {
	positive := []struct {
		name  string
		value *int
	}{
		{
			name:  "inputWidth",
			value: o.InputWidth,
		},
		{
			name:  "inputHeight",
			value: o.InputHeight,
		},
		{
			name:  "pollIntervalMillis",
			value: o.PollIntervalMillis,
		},
		{
			name:  "pollTimeoutMillis",
			value: o.PollTimeoutMillis,
		},
	}

	for _, check := range positive {
		if check.value != nil && *check.value <= 0 {
			return fmt.Errorf("%w: topaz %s must be positive", model.ErrInvalidArgument, check.name)
		}
	}

	if o.OutputWidth != nil && (*o.OutputWidth < 1 || *o.OutputWidth > 32000) {
		return fmt.Errorf("%w: topaz outputWidth must be between 1 and 32000", model.ErrInvalidArgument)
	}

	if o.OutputHeight != nil && (*o.OutputHeight < 1 || *o.OutputHeight > 32000) {
		return fmt.Errorf("%w: topaz outputHeight must be between 1 and 32000", model.ErrInvalidArgument)
	}

	return nil
}
