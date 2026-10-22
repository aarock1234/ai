package luma

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// ReferenceType selects how input images guide a generation.
type ReferenceType string

// Reference types.
const (
	// ReferenceImage guides generation with up to four reference images. It
	// is the default.
	ReferenceImage ReferenceType = "image"

	// ReferenceStyle applies the style of the reference images.
	ReferenceStyle ReferenceType = "style"

	// ReferenceCharacter creates consistent characters from up to four
	// images per identity.
	ReferenceCharacter ReferenceType = "character"

	// ReferenceModifyImage transforms a single input image with the prompt.
	ReferenceModifyImage ReferenceType = "modify_image"
)

// ImageReference configures one input image. Entries correspond to the
// input files by position.
type ImageReference struct {
	// Weight is the image's influence on the generation, from 0 to 1. The
	// default is 0.85 for image references, 0.8 for style references and 1
	// for modify_image. It does not apply to character references.
	Weight *float64 `json:"weight,omitempty"`

	// ID is the identity group of a character reference. The default is
	// "identity0".
	ID *string `json:"id,omitempty"`
}

// ImageOptions are the provider options of image models, read from the
// "luma" namespace. Members other than these fields are sent in the request
// body unchanged.
type ImageOptions struct {
	// ReferenceType selects how input images are used. The default is
	// ReferenceImage.
	ReferenceType *ReferenceType `json:"referenceType,omitempty"`

	// Images configures each input image.
	Images []ImageReference `json:"images,omitempty"`

	// PollIntervalMillis is the delay between status checks. The default is
	// 500.
	PollIntervalMillis *int `json:"pollIntervalMillis,omitempty"`

	// MaxPollAttempts limits the status checks. The default is 120.
	MaxPollAttempts *int `json:"maxPollAttempts,omitempty"`
}

// imageOptionKeys are the ImageOptions members, which are not sent in the
// request body.
var imageOptionKeys = []string{
	"referenceType",
	"images",
	"pollIntervalMillis",
	"maxPollAttempts",
}

// readImageOptions reads and validates the call's Luma options and returns
// the members to pass through to the request body.
func readImageOptions(options model.ProviderOptions) (ImageOptions, map[string]json.RawMessage, error) {
	imageOptions, _, err := model.ProviderValue[ImageOptions](options, Namespace)
	if err != nil {
		return ImageOptions{}, nil, err
	}

	if err := imageOptions.validate(); err != nil {
		return ImageOptions{}, nil, err
	}

	passthrough, _, err := model.ProviderValue[map[string]json.RawMessage](options, Namespace)
	if err != nil {
		return ImageOptions{}, nil, err
	}

	for _, key := range imageOptionKeys {
		delete(passthrough, key)
	}

	return imageOptions, passthrough, nil
}

func (o ImageOptions) validate() error {
	if o.ReferenceType != nil {
		switch *o.ReferenceType {
		case ReferenceImage, ReferenceStyle, ReferenceCharacter, ReferenceModifyImage:
		default:
			return fmt.Errorf("%w: luma referenceType %q", model.ErrInvalidArgument, *o.ReferenceType)
		}
	}

	for index, image := range o.Images {
		if image.Weight != nil && (*image.Weight < 0 || *image.Weight > 1) {
			return fmt.Errorf("%w: luma images[%d].weight must be between 0 and 1", model.ErrInvalidArgument, index)
		}
	}

	return nil
}
