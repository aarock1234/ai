package fal

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

// ImageOutputFormat is the file format of generated images.
type ImageOutputFormat string

// Image output formats.
const (
	ImageOutputJPEG ImageOutputFormat = "jpeg"
	ImageOutputPNG  ImageOutputFormat = "png"
)

// Acceleration selects the inference speed-up.
type Acceleration string

// Accelerations.
const (
	AccelerationNone    Acceleration = "none"
	AccelerationRegular Acceleration = "regular"
	AccelerationHigh    Acceleration = "high"
)

// SafetyTolerance is the moderation tolerance, from 1 (strictest) to 6. It
// accepts a JSON number or a string such as "2".
type SafetyTolerance float64

// UnmarshalJSON reads a number from 1 to 6 or one of the strings "1" to "6".
func (t *SafetyTolerance) UnmarshalJSON(data []byte) error {
	var text string
	if err := json.Unmarshal(data, &text); err == nil {
		level, err := strconv.Atoi(text)
		if err != nil || level < 1 || level > 6 {
			return fmt.Errorf("safety tolerance %q must be one of \"1\" to \"6\"", text)
		}

		*t = SafetyTolerance(level)

		return nil
	}

	var level float64
	if err := json.Unmarshal(data, &level); err != nil {
		return fmt.Errorf("safety tolerance must be a number or a string: %w", err)
	}

	if level < 1 || level > 6 {
		return fmt.Errorf("safety tolerance %v must be between 1 and 6", level)
	}

	*t = SafetyTolerance(level)

	return nil
}

// ImageOptions are the provider options of image models, read from the
// "fal" namespace. Known options are sent in snake_case; other members are
// sent unchanged, so model-specific inputs can be passed directly. The
// snake_case spellings of the known options are deprecated but accepted
// with a warning.
type ImageOptions struct {
	// ImageURL is deprecated; use the input files of the call instead.
	ImageURL *string `json:"imageUrl,omitempty"`

	// MaskURL is deprecated; use the mask of the call instead.
	MaskURL *string `json:"maskUrl,omitempty"`

	// GuidanceScale is the classifier-free guidance scale, from 1 to 20.
	GuidanceScale *float64 `json:"guidanceScale,omitempty"`

	// NumInferenceSteps is the number of denoising steps, from 1 to 50.
	NumInferenceSteps *float64 `json:"numInferenceSteps,omitempty"`

	// EnableSafetyChecker enables the safety checker.
	EnableSafetyChecker *bool `json:"enableSafetyChecker,omitempty"`

	// OutputFormat is the image file format.
	OutputFormat *ImageOutputFormat `json:"outputFormat,omitempty"`

	// SyncMode waits for the image to be uploaded before responding.
	SyncMode *bool `json:"syncMode,omitempty"`

	// Strength is the image-to-image transformation strength.
	Strength *float64 `json:"strength,omitempty"`

	// Acceleration selects the inference speed-up.
	Acceleration *Acceleration `json:"acceleration,omitempty"`

	// SafetyTolerance is the moderation tolerance.
	SafetyTolerance *SafetyTolerance `json:"safetyTolerance,omitempty"`

	// UseMultipleImages sends every input file as "image_urls" instead of
	// sending the first as "image_url". Models such as fal-ai/flux-2/edit
	// need it.
	UseMultipleImages *bool `json:"useMultipleImages,omitempty"`
}

// deprecatedImageOptions are the deprecated snake_case spellings.
type deprecatedImageOptions struct {
	ImageURL            *string            `json:"image_url"`
	MaskURL             *string            `json:"mask_url"`
	GuidanceScale       *float64           `json:"guidance_scale"`
	NumInferenceSteps   *float64           `json:"num_inference_steps"`
	EnableSafetyChecker *bool              `json:"enable_safety_checker"`
	OutputFormat        *ImageOutputFormat `json:"output_format"`
	SyncMode            *bool              `json:"sync_mode"`
	SafetyTolerance     *SafetyTolerance   `json:"safety_tolerance"`
}

// imageOptionName pairs an option's camelCase and snake_case spellings.
// The snake_case spelling is also the request field.
type imageOptionName struct {
	camel string
	snake string
}

// mappedImageOptions are the options with a deprecated snake_case spelling.
var mappedImageOptions = []imageOptionName{
	{
		camel: "imageUrl",
		snake: "image_url",
	},
	{
		camel: "maskUrl",
		snake: "mask_url",
	},
	{
		camel: "guidanceScale",
		snake: "guidance_scale",
	},
	{
		camel: "numInferenceSteps",
		snake: "num_inference_steps",
	},
	{
		camel: "enableSafetyChecker",
		snake: "enable_safety_checker",
	},
	{
		camel: "outputFormat",
		snake: "output_format",
	},
	{
		camel: "syncMode",
		snake: "sync_mode",
	},
	{
		camel: "safetyTolerance",
		snake: "safety_tolerance",
	},
}

// unmappedImageOptions are the known options sent under their own name.
var unmappedImageOptions = []string{
	"strength",
	"acceleration",
}

// imageRequestOptions are the request members derived from the provider
// options.
type imageRequestOptions struct {
	members           map[string]json.RawMessage
	deprecatedKeys    []string
	useMultipleImages bool
}

// readImageOptions validates the call's fal options and maps them onto
// request members.
func readImageOptions(options model.ProviderOptions) (imageRequestOptions, error) {
	imageOptions, _, err := model.ProviderValue[ImageOptions](options, Namespace)
	if err != nil {
		return imageRequestOptions{}, err
	}

	deprecated, _, err := model.ProviderValue[deprecatedImageOptions](options, Namespace)
	if err != nil {
		return imageRequestOptions{}, err
	}

	if err := imageOptions.validate(); err != nil {
		return imageRequestOptions{}, err
	}

	if err := deprecated.validate(); err != nil {
		return imageRequestOptions{}, err
	}

	raw, _, err := model.ProviderValue[map[string]json.RawMessage](options, Namespace)
	if err != nil {
		return imageRequestOptions{}, err
	}

	return mapImageOptions(raw, imageOptions), nil
}

// mapImageOptions converts the options to request members. A deprecated
// spelling wins over the camelCase one; known options without a value are
// dropped, and unknown members are sent unchanged.
func mapImageOptions(raw map[string]json.RawMessage, imageOptions ImageOptions) imageRequestOptions {
	result := imageRequestOptions{
		members:           make(map[string]json.RawMessage),
		useMultipleImages: imageOptions.UseMultipleImages != nil && *imageOptions.UseMultipleImages,
	}

	known := []string{
		"useMultipleImages",
	}

	for _, name := range mappedImageOptions {
		known = append(known, name.camel, name.snake)

		if value, ok := present(raw, name.snake); ok {
			result.deprecatedKeys = append(result.deprecatedKeys, name.snake)
			result.members[name.snake] = value
		} else if value, ok := present(raw, name.camel); ok {
			result.members[name.snake] = value
		}
	}

	for _, key := range unmappedImageOptions {
		known = append(known, key)

		if value, ok := present(raw, key); ok {
			result.members[key] = value
		}
	}

	for key, value := range raw {
		if !slices.Contains(known, key) {
			result.members[key] = value
		}
	}

	return result
}

// present returns the member named key unless it is absent or null.
func present(raw map[string]json.RawMessage, key string) (json.RawMessage, bool) {
	value, ok := raw[key]
	if !ok || bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
		return nil, false
	}

	return value, true
}

// deprecationWarning lists the deprecated snake_case options with their
// replacements.
func deprecationWarning(keys []string) model.Warning {
	descriptions := make([]string, 0, len(keys))
	for _, key := range keys {
		descriptions = append(descriptions, fmt.Sprintf("'%s' (use '%s')", key, snakeToCamel(key)))
	}

	return model.Warning{
		Type: model.WarningOther,
		Message: "The following provider options use deprecated snake_case and will be removed in @ai-sdk/fal v2.0. " +
			"Please use camelCase instead: " + strings.Join(descriptions, ", "),
	}
}

func snakeToCamel(key string) string {
	words := strings.Split(key, "_")
	for index := 1; index < len(words); index++ {
		if words[index] != "" {
			words[index] = strings.ToUpper(words[index][:1]) + words[index][1:]
		}
	}

	return strings.Join(words, "")
}

func (o ImageOptions) validate() error {
	if err := validateImageSettings(o.GuidanceScale, o.NumInferenceSteps, o.OutputFormat); err != nil {
		return err
	}

	if o.Acceleration != nil {
		switch *o.Acceleration {
		case AccelerationNone, AccelerationRegular, AccelerationHigh:
		default:
			return fmt.Errorf("%w: fal acceleration %q", model.ErrInvalidArgument, *o.Acceleration)
		}
	}

	return nil
}

func (o deprecatedImageOptions) validate() error {
	return validateImageSettings(o.GuidanceScale, o.NumInferenceSteps, o.OutputFormat)
}

func validateImageSettings(guidanceScale *float64, numInferenceSteps *float64, outputFormat *ImageOutputFormat) error {
	if guidanceScale != nil && (*guidanceScale < 1 || *guidanceScale > 20) {
		return fmt.Errorf("%w: fal guidanceScale must be between 1 and 20", model.ErrInvalidArgument)
	}

	if numInferenceSteps != nil && (*numInferenceSteps < 1 || *numInferenceSteps > 50) {
		return fmt.Errorf("%w: fal numInferenceSteps must be between 1 and 50", model.ErrInvalidArgument)
	}

	if outputFormat != nil && *outputFormat != ImageOutputJPEG && *outputFormat != ImageOutputPNG {
		return fmt.Errorf("%w: fal outputFormat %q", model.ErrInvalidArgument, *outputFormat)
	}

	return nil
}
