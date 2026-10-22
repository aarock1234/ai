package blackforestlabs

import (
	"fmt"
	"net/url"

	"github.com/aarock1234/ai/pkg/model"
)

// maxLatestPixels is the largest output area of the latest image API.
const maxLatestPixels = 16822272

// Resolution is an output resolution of FLUX 3 and later models.
type Resolution string

// Resolutions.
const (
	Resolution768Square Resolution = "768sq"
	Resolution1K        Resolution = "1k"
	Resolution1_5K      Resolution = "1.5k"
	Resolution2K        Resolution = "2k"
	Resolution4K        Resolution = "4k"
)

// OutputFormat is the file format of generated images.
type OutputFormat string

// Output formats.
const (
	OutputJPEG OutputFormat = "jpeg"
	OutputPNG  OutputFormat = "png"
)

// ImageOptions are the provider options of image models, read from the
// "blackForestLabs" namespace. Models support different subsets: FLUX 3 and
// later models report unsupported options as warnings.
type ImageOptions struct {
	// Resolution selects the output resolution of FLUX 3 and later models.
	Resolution *Resolution `json:"resolution,omitempty"`

	// Grounding enables grounding on flux-3-image.
	Grounding *bool `json:"grounding,omitempty"`

	// Version selects "latest" on flux-3-image.
	Version *string `json:"version,omitempty"`

	// ImagePrompt is a base64 image that guides generation on legacy models.
	ImagePrompt *string `json:"imagePrompt,omitempty"`

	// ImagePromptStrength is the influence of ImagePrompt, from 0 to 1.
	ImagePromptStrength *float64 `json:"imagePromptStrength,omitempty"`

	// InputImage through InputImage10 are deprecated; use the input files of
	// the call instead.
	InputImage   *string `json:"inputImage,omitempty"`
	InputImage2  *string `json:"inputImage2,omitempty"`
	InputImage3  *string `json:"inputImage3,omitempty"`
	InputImage4  *string `json:"inputImage4,omitempty"`
	InputImage5  *string `json:"inputImage5,omitempty"`
	InputImage6  *string `json:"inputImage6,omitempty"`
	InputImage7  *string `json:"inputImage7,omitempty"`
	InputImage8  *string `json:"inputImage8,omitempty"`
	InputImage9  *string `json:"inputImage9,omitempty"`
	InputImage10 *string `json:"inputImage10,omitempty"`

	// Steps is the number of generation steps.
	Steps *int `json:"steps,omitempty"`

	// Guidance is the guidance scale, at least 0.
	Guidance *float64 `json:"guidance,omitempty"`

	// Width and Height are the output dimensions in pixels. Legacy models
	// accept 256 to 1920; the latest API accepts 64 to 6256 and requires
	// both together.
	Width  *int `json:"width,omitempty"`
	Height *int `json:"height,omitempty"`

	// OutputFormat is the image file format.
	OutputFormat *OutputFormat `json:"outputFormat,omitempty"`

	// PromptUpsampling rewrites the prompt for more detail.
	PromptUpsampling *bool `json:"promptUpsampling,omitempty"`

	// Raw requests less processed, more natural images on ultra models.
	Raw *bool `json:"raw,omitempty"`

	// SafetyTolerance is the moderation tolerance: 0 to 6 on legacy models
	// and 0 to 7 on FLUX 3 and later models.
	SafetyTolerance *int `json:"safetyTolerance,omitempty"`

	// WebhookSecret signs webhook deliveries.
	WebhookSecret *string `json:"webhookSecret,omitempty"`

	// WebhookURL receives the result when it is ready.
	WebhookURL *string `json:"webhookUrl,omitempty"`

	// PollIntervalMillis overrides the provider's poll interval.
	PollIntervalMillis *int `json:"pollIntervalMillis,omitempty"`

	// PollTimeoutMillis overrides the provider's poll timeout.
	PollTimeoutMillis *int `json:"pollTimeoutMillis,omitempty"`
}

// optionPresence is one option and whether the call set it.
type optionPresence struct {
	key string
	set bool
}

// presence lists the options in declaration order with whether each is set,
// so unsupported options are reported in a stable order.
func (o ImageOptions) presence() []optionPresence {
	return []optionPresence{
		{
			key: "resolution",
			set: o.Resolution != nil,
		},
		{
			key: "grounding",
			set: o.Grounding != nil,
		},
		{
			key: "version",
			set: o.Version != nil,
		},
		{
			key: "imagePrompt",
			set: o.ImagePrompt != nil,
		},
		{
			key: "imagePromptStrength",
			set: o.ImagePromptStrength != nil,
		},
		{
			key: "inputImage",
			set: o.InputImage != nil,
		},
		{
			key: "inputImage2",
			set: o.InputImage2 != nil,
		},
		{
			key: "inputImage3",
			set: o.InputImage3 != nil,
		},
		{
			key: "inputImage4",
			set: o.InputImage4 != nil,
		},
		{
			key: "inputImage5",
			set: o.InputImage5 != nil,
		},
		{
			key: "inputImage6",
			set: o.InputImage6 != nil,
		},
		{
			key: "inputImage7",
			set: o.InputImage7 != nil,
		},
		{
			key: "inputImage8",
			set: o.InputImage8 != nil,
		},
		{
			key: "inputImage9",
			set: o.InputImage9 != nil,
		},
		{
			key: "inputImage10",
			set: o.InputImage10 != nil,
		},
		{
			key: "steps",
			set: o.Steps != nil,
		},
		{
			key: "guidance",
			set: o.Guidance != nil,
		},
		{
			key: "width",
			set: o.Width != nil,
		},
		{
			key: "height",
			set: o.Height != nil,
		},
		{
			key: "outputFormat",
			set: o.OutputFormat != nil,
		},
		{
			key: "promptUpsampling",
			set: o.PromptUpsampling != nil,
		},
		{
			key: "raw",
			set: o.Raw != nil,
		},
		{
			key: "safetyTolerance",
			set: o.SafetyTolerance != nil,
		},
		{
			key: "webhookSecret",
			set: o.WebhookSecret != nil,
		},
		{
			key: "webhookUrl",
			set: o.WebhookURL != nil,
		},
		{
			key: "pollIntervalMillis",
			set: o.PollIntervalMillis != nil,
		},
		{
			key: "pollTimeoutMillis",
			set: o.PollTimeoutMillis != nil,
		},
	}
}

// readImageOptions reads the call's options and validates them against the
// model's API. For the latest API, a size sets the default width and height.
func readImageOptions(options model.ProviderOptions, api modelAPI, size *imageSize) (ImageOptions, error) {
	imageOptions, _, err := model.ProviderValue[ImageOptions](options, Namespace)
	if err != nil {
		return ImageOptions{}, err
	}

	if api == apiLatest && size != nil {
		if imageOptions.Width == nil {
			imageOptions.Width = new(size.width)
		}

		if imageOptions.Height == nil {
			imageOptions.Height = new(size.height)
		}
	}

	if err := imageOptions.validate(api); err != nil {
		return ImageOptions{}, err
	}

	return imageOptions, nil
}

func (o ImageOptions) validate(api modelAPI) error {
	if err := o.validateCommon(); err != nil {
		return err
	}

	switch api {
	case apiLatest:
		return o.validateLatest()

	case apiFlux3:
		if err := checkIntRange("width", o.Width, 256, 1920); err != nil {
			return err
		}

		if err := checkIntRange("height", o.Height, 256, 1920); err != nil {
			return err
		}

		return checkIntRange("safetyTolerance", o.SafetyTolerance, 0, 7)

	default:
		if err := checkIntRange("width", o.Width, 256, 1920); err != nil {
			return err
		}

		if err := checkIntRange("height", o.Height, 256, 1920); err != nil {
			return err
		}

		return checkIntRange("safetyTolerance", o.SafetyTolerance, 0, 6)
	}
}

// validateCommon checks the constraints shared by every API.
func (o ImageOptions) validateCommon() error {
	if o.Resolution != nil {
		switch *o.Resolution {
		case Resolution768Square, Resolution1K, Resolution1_5K, Resolution2K, Resolution4K:
		default:
			return fmt.Errorf("%w: blackForestLabs resolution %q", model.ErrInvalidArgument, *o.Resolution)
		}
	}

	if o.Version != nil && *o.Version != "latest" {
		return fmt.Errorf("%w: blackForestLabs version must be \"latest\"", model.ErrInvalidArgument)
	}

	if o.ImagePromptStrength != nil && (*o.ImagePromptStrength < 0 || *o.ImagePromptStrength > 1) {
		return fmt.Errorf("%w: blackForestLabs imagePromptStrength must be between 0 and 1", model.ErrInvalidArgument)
	}

	if o.Steps != nil && *o.Steps <= 0 {
		return fmt.Errorf("%w: blackForestLabs steps must be positive", model.ErrInvalidArgument)
	}

	if o.Guidance != nil && *o.Guidance < 0 {
		return fmt.Errorf("%w: blackForestLabs guidance must not be negative", model.ErrInvalidArgument)
	}

	if o.OutputFormat != nil && *o.OutputFormat != OutputJPEG && *o.OutputFormat != OutputPNG {
		return fmt.Errorf("%w: blackForestLabs outputFormat %q", model.ErrInvalidArgument, *o.OutputFormat)
	}

	if o.WebhookURL != nil {
		if _, err := url.ParseRequestURI(*o.WebhookURL); err != nil {
			return fmt.Errorf("%w: blackForestLabs webhookUrl: %w", model.ErrInvalidArgument, err)
		}
	}

	if o.PollIntervalMillis != nil && *o.PollIntervalMillis <= 0 {
		return fmt.Errorf("%w: blackForestLabs pollIntervalMillis must be positive", model.ErrInvalidArgument)
	}

	if o.PollTimeoutMillis != nil && *o.PollTimeoutMillis <= 0 {
		return fmt.Errorf("%w: blackForestLabs pollTimeoutMillis must be positive", model.ErrInvalidArgument)
	}

	return nil
}

// validateLatest checks the dimensions, steps and safety tolerance of the
// latest image API.
func (o ImageOptions) validateLatest() error {
	if err := checkIntRange("width", o.Width, 64, 6256); err != nil {
		return err
	}

	if err := checkIntRange("height", o.Height, 64, 6256); err != nil {
		return err
	}

	if err := checkIntRange("steps", o.Steps, 1, 16); err != nil {
		return err
	}

	if err := checkIntRange("safetyTolerance", o.SafetyTolerance, 0, 7); err != nil {
		return err
	}

	if (o.Width == nil) != (o.Height == nil) {
		return fmt.Errorf("%w: blackForestLabs width and height must be provided together", model.ErrInvalidArgument)
	}

	if o.Width != nil && *o.Width**o.Height > maxLatestPixels {
		return fmt.Errorf("%w: blackForestLabs width × height must not exceed %d pixels", model.ErrInvalidArgument, maxLatestPixels)
	}

	return nil
}

func checkIntRange(name string, value *int, minimum int, maximum int) error {
	if value == nil || (*value >= minimum && *value <= maximum) {
		return nil
	}

	return fmt.Errorf("%w: blackForestLabs %s must be between %d and %d", model.ErrInvalidArgument, name, minimum, maximum)
}
