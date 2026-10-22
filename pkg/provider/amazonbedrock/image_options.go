package amazonbedrock

import (
	"github.com/aarock1234/ai/pkg/model"
)

// ImageQuality is the Nova Canvas image quality.
type ImageQuality string

// Image qualities.
const (
	ImageQualityStandard ImageQuality = "standard"
	ImageQualityPremium  ImageQuality = "premium"
)

// ImageStyle is a Nova Canvas visual style for text-to-image generation.
type ImageStyle string

// Image styles.
const (
	ImageStyle3DAnimatedFamilyFilm     ImageStyle = "3D_ANIMATED_FAMILY_FILM"
	ImageStyleDesignSketch             ImageStyle = "DESIGN_SKETCH"
	ImageStyleFlatVectorIllustration   ImageStyle = "FLAT_VECTOR_ILLUSTRATION"
	ImageStyleGraphicNovelIllustration ImageStyle = "GRAPHIC_NOVEL_ILLUSTRATION"
	ImageStyleMaximalism               ImageStyle = "MAXIMALISM"
	ImageStyleMidcenturyRetro          ImageStyle = "MIDCENTURY_RETRO"
	ImageStylePhotorealism             ImageStyle = "PHOTOREALISM"
	ImageStyleSoftDigitalPainting      ImageStyle = "SOFT_DIGITAL_PAINTING"
)

// ImageTaskType is a Nova Canvas task.
type ImageTaskType string

// Image tasks.
const (
	ImageTaskTextImage         ImageTaskType = "TEXT_IMAGE"
	ImageTaskImageVariation    ImageTaskType = "IMAGE_VARIATION"
	ImageTaskInpainting        ImageTaskType = "INPAINTING"
	ImageTaskOutpainting       ImageTaskType = "OUTPAINTING"
	ImageTaskBackgroundRemoval ImageTaskType = "BACKGROUND_REMOVAL"
)

// OutPaintingMode selects how strictly outpainting follows the mask.
type OutPaintingMode string

// Outpainting modes.
const (
	OutPaintingDefault OutPaintingMode = "DEFAULT"
	OutPaintingPrecise OutPaintingMode = "PRECISE"
)

// ImageOptions are the provider options of image models, read from the
// "amazonBedrock" namespace, or from "bedrock" when the call has no
// "amazonBedrock" options.
type ImageOptions struct {
	Quality *ImageQuality `json:"quality,omitempty"`

	// NegativeText describes what the image should not contain.
	NegativeText *string `json:"negativeText,omitempty"`

	// CfgScale sets how strictly the image follows the prompt.
	CfgScale *float64 `json:"cfgScale,omitempty"`

	// Style applies to text-to-image generation.
	Style *ImageStyle `json:"style,omitempty"`

	// TaskType selects the editing task when the call has input images.
	// The default is inpainting with a mask or mask prompt, and image
	// variation otherwise.
	TaskType *ImageTaskType `json:"taskType,omitempty"`

	// MaskPrompt describes the area to edit instead of a mask image.
	MaskPrompt *string `json:"maskPrompt,omitempty"`

	OutPaintingMode *OutPaintingMode `json:"outPaintingMode,omitempty"`

	// SimilarityStrength sets how similar image variations stay to their
	// sources.
	SimilarityStrength *float64 `json:"similarityStrength,omitempty"`
}

func readImageOptions(options model.ProviderOptions) (ImageOptions, error) {
	value, _, err := namespacedValue[ImageOptions](options)
	if err != nil {
		return ImageOptions{}, err
	}

	if err := value.validate(); err != nil {
		return ImageOptions{}, err
	}

	return value, nil
}

func (o ImageOptions) validate() error {
	if o.Quality != nil {
		qualities := []ImageQuality{
			ImageQualityStandard,
			ImageQualityPremium,
		}

		if err := checkEnum("quality", *o.Quality, qualities); err != nil {
			return err
		}
	}

	if o.Style != nil {
		styles := []ImageStyle{
			ImageStyle3DAnimatedFamilyFilm,
			ImageStyleDesignSketch,
			ImageStyleFlatVectorIllustration,
			ImageStyleGraphicNovelIllustration,
			ImageStyleMaximalism,
			ImageStyleMidcenturyRetro,
			ImageStylePhotorealism,
			ImageStyleSoftDigitalPainting,
		}

		if err := checkEnum("style", *o.Style, styles); err != nil {
			return err
		}
	}

	if o.TaskType != nil {
		tasks := []ImageTaskType{
			ImageTaskTextImage,
			ImageTaskImageVariation,
			ImageTaskInpainting,
			ImageTaskOutpainting,
			ImageTaskBackgroundRemoval,
		}

		if err := checkEnum("taskType", *o.TaskType, tasks); err != nil {
			return err
		}
	}

	if o.OutPaintingMode != nil {
		modes := []OutPaintingMode{
			OutPaintingDefault,
			OutPaintingPrecise,
		}

		return checkEnum("outPaintingMode", *o.OutPaintingMode, modes)
	}

	return nil
}
