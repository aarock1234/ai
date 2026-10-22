package quiverai

import (
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// maxReferenceImages is the reference image limit of SVG editing.
const maxReferenceImages = 4

// Operation selects what an image call does.
type Operation string

// Operations.
const (
	// OperationGenerate generates SVGs from the prompt. It is the default.
	OperationGenerate Operation = "generate"

	// OperationVectorize converts one input raster image into an SVG.
	OperationVectorize Operation = "vectorize"

	// OperationAnimate animates one input SVG; the prompt is optional.
	OperationAnimate Operation = "animate"

	// OperationEdit edits one input SVG following the prompt.
	OperationEdit Operation = "edit"
)

// ReasoningEffort is the reasoning effort of generation and vectorization.
type ReasoningEffort string

// Reasoning efforts.
const (
	ReasoningLow    ReasoningEffort = "low"
	ReasoningMedium ReasoningEffort = "medium"
	ReasoningHigh   ReasoningEffort = "high"
	ReasoningXHigh  ReasoningEffort = "xhigh"
)

// Attributes are SVG root attributes requested for generation or
// vectorization.
type Attributes struct {
	ViewBox *ViewBox `json:"viewBox,omitempty"`
}

// ViewBox is the SVG viewBox attribute.
type ViewBox struct {
	MinX   float64 `json:"minX"`
	MinY   float64 `json:"minY"`
	Width  float64 `json:"width"`
	Height float64 `json:"height"`
}

// ImageOptions are the provider options of image models, read from the
// "quiverai" namespace. Operations accept different subsets and reject the
// others.
type ImageOptions struct {
	// Operation selects the operation. The default is OperationGenerate.
	Operation *Operation `json:"operation,omitempty"`

	// Instructions add style guidance to prompt-based generation.
	Instructions *string `json:"instructions,omitempty"`

	// ReasoningEffort is the reasoning effort.
	ReasoningEffort *ReasoningEffort `json:"reasoningEffort,omitempty"`

	// ReferenceImages are up to four reference images for SVG editing. Use
	// PrepareImageReference to build them.
	ReferenceImages []ImageReference `json:"referenceImages,omitempty"`

	// MaxReviewSteps limits the edit review and redo steps, from 0 to 5.
	MaxReviewSteps *int `json:"maxReviewSteps,omitempty"`

	// Attributes are the SVG root attributes for generation or
	// vectorization.
	Attributes *Attributes `json:"attributes,omitempty"`

	// Temperature is the sampling temperature, from 0 to 2.
	Temperature *float64 `json:"temperature,omitempty"`

	// TopP is the nucleus sampling probability, from 0 to 1.
	TopP *float64 `json:"topP,omitempty"`

	// PresencePenalty is the presence penalty, from -2 to 2.
	PresencePenalty *float64 `json:"presencePenalty,omitempty"`

	// MaxOutputTokens limits the output tokens: up to 65536 for Arrow 2
	// models and 131072 otherwise.
	MaxOutputTokens *int `json:"maxOutputTokens,omitempty"`

	// OrchestratorMaxOutputTokens is the orchestrator token budget of SVG
	// editing, from 1 to 65536.
	OrchestratorMaxOutputTokens *int `json:"orchestratorMaxOutputTokens,omitempty"`

	// ShallowMaxOutputTokens is the shallow edit token budget of SVG
	// editing, from 1 to 65536.
	ShallowMaxOutputTokens *int `json:"shallowMaxOutputTokens,omitempty"`

	// AutoCrop crops the input image before vectorization.
	AutoCrop *bool `json:"autoCrop,omitempty"`

	// TargetSize is the vectorization canvas size, from 128 to 4096 pixels.
	TargetSize *int `json:"targetSize,omitempty"`
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
	if err := o.validateChoices(); err != nil {
		return err
	}

	if err := o.validateReferences(); err != nil {
		return err
	}

	if o.Attributes != nil && o.Attributes.ViewBox != nil {
		if o.Attributes.ViewBox.Width <= 0 || o.Attributes.ViewBox.Height <= 0 {
			return fmt.Errorf("%w: quiverai attributes.viewBox width and height must be positive", model.ErrInvalidArgument)
		}
	}

	return o.validateRanges()
}

func (o ImageOptions) validateChoices() error {
	if o.Operation != nil {
		switch *o.Operation {
		case OperationGenerate, OperationVectorize, OperationAnimate, OperationEdit:
		default:
			return fmt.Errorf("%w: quiverai operation %q", model.ErrInvalidArgument, *o.Operation)
		}
	}

	if o.ReasoningEffort != nil {
		switch *o.ReasoningEffort {
		case ReasoningLow, ReasoningMedium, ReasoningHigh, ReasoningXHigh:
		default:
			return fmt.Errorf("%w: quiverai reasoningEffort %q", model.ErrInvalidArgument, *o.ReasoningEffort)
		}
	}

	if o.Instructions != nil && *o.Instructions == "" {
		return fmt.Errorf("%w: quiverai instructions must not be empty", model.ErrInvalidArgument)
	}

	return nil
}

func (o ImageOptions) validateReferences() error {
	if len(o.ReferenceImages) > maxReferenceImages {
		return fmt.Errorf("%w: quiverai accepts at most %d referenceImages", model.ErrInvalidArgument, maxReferenceImages)
	}

	for index, reference := range o.ReferenceImages {
		hasURL := reference.URL != ""
		hasBase64 := reference.Base64 != ""

		if hasURL == hasBase64 {
			return fmt.Errorf("%w: quiverai referenceImages[%d] must have exactly one of url and base64", model.ErrInvalidArgument, index)
		}

		if len(reference.Base64) > maxReferenceBase64Length {
			return fmt.Errorf(
				"%w: quiverai referenceImages[%d] must contain at most %d base64 characters",
				model.ErrInvalidArgument,
				index,
				maxReferenceBase64Length,
			)
		}
	}

	return nil
}

func (o ImageOptions) validateRanges() error {
	floatRanges := []struct {
		name    string
		value   *float64
		minimum float64
		maximum float64
	}{
		{
			name:    "temperature",
			value:   o.Temperature,
			minimum: 0,
			maximum: 2,
		},
		{
			name:    "topP",
			value:   o.TopP,
			minimum: 0,
			maximum: 1,
		},
		{
			name:    "presencePenalty",
			value:   o.PresencePenalty,
			minimum: -2,
			maximum: 2,
		},
	}

	for _, check := range floatRanges {
		if check.value != nil && (*check.value < check.minimum || *check.value > check.maximum) {
			return fmt.Errorf("%w: quiverai %s must be between %v and %v", model.ErrInvalidArgument, check.name, check.minimum, check.maximum)
		}
	}

	intRanges := []struct {
		name    string
		value   *int
		minimum int
		maximum int
	}{
		{
			name:    "maxReviewSteps",
			value:   o.MaxReviewSteps,
			minimum: 0,
			maximum: 5,
		},
		{
			name:    "maxOutputTokens",
			value:   o.MaxOutputTokens,
			minimum: 1,
			maximum: 131072,
		},
		{
			name:    "orchestratorMaxOutputTokens",
			value:   o.OrchestratorMaxOutputTokens,
			minimum: 1,
			maximum: 65536,
		},
		{
			name:    "shallowMaxOutputTokens",
			value:   o.ShallowMaxOutputTokens,
			minimum: 1,
			maximum: 65536,
		},
		{
			name:    "targetSize",
			value:   o.TargetSize,
			minimum: 128,
			maximum: 4096,
		},
	}

	for _, check := range intRanges {
		if check.value != nil && (*check.value < check.minimum || *check.value > check.maximum) {
			return fmt.Errorf("%w: quiverai %s must be between %d and %d", model.ErrInvalidArgument, check.name, check.minimum, check.maximum)
		}
	}

	return nil
}
