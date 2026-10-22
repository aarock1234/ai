package heygen

import (
	"fmt"
	"net/url"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

// Reference limits of one generation.
const (
	maxReferenceImages = 9
	maxReferenceVideos = 3
	maxReferenceAudio  = 3
	maxReferences      = 12
)

// VideoMode is the generation mode.
type VideoMode string

// Generation modes.
const (
	ModeTextToVideo      VideoMode = "text_to_video"
	ModeImageToVideo     VideoMode = "image_to_video"
	ModeReferenceToVideo VideoMode = "reference_to_video"
)

// VideoResolution is an output size class.
type VideoResolution string

// Size classes.
const (
	Resolution480p  VideoResolution = "480p"
	Resolution768p  VideoResolution = "768p"
	Resolution1080p VideoResolution = "1080p"
	Resolution2K    VideoResolution = "2k"
)

// PromptEnhancement controls prompt rewriting.
type PromptEnhancement string

// Prompt enhancement settings.
const (
	PromptEnhancementTurbo    PromptEnhancement = "turbo"
	PromptEnhancementQuality  PromptEnhancement = "quality"
	PromptEnhancementDisabled PromptEnhancement = "disabled"
)

// VideoAssetType is the kind of a VideoAsset.
type VideoAssetType string

// Asset kinds.
const (
	// AssetURL is an absolute HTTPS URL.
	AssetURL VideoAssetType = "url"

	// AssetID is an existing HeyGen asset.
	AssetID VideoAssetType = "asset_id"

	// AssetBase64 is inline base64 data with its media type.
	AssetBase64 VideoAssetType = "base64"
)

// VideoAsset is an input image, video or audio file. Type selects the
// fields it uses: URL, AssetID, or MediaType and Data.
type VideoAsset struct {
	Type      VideoAssetType `json:"type"`
	URL       string         `json:"url,omitempty"`
	AssetID   string         `json:"assetId,omitempty"`
	MediaType string         `json:"mediaType,omitempty"`
	Data      string         `json:"data,omitempty"`
}

// VideoOptions are the provider options of video models, read from the
// "heygen" namespace.
type VideoOptions struct {
	// Mode is inferred from the image and references when nil.
	Mode *VideoMode `json:"mode,omitempty"`

	// Resolution is the output size class. It takes precedence over the
	// standard resolution option.
	Resolution *VideoResolution `json:"resolution,omitempty"`

	// PromptEnhancement defaults to turbo. Disabled keeps the prompt.
	PromptEnhancement *PromptEnhancement `json:"promptEnhancement,omitempty"`

	// Image is the first frame, including an existing HeyGen asset. It
	// cannot be combined with a standard image.
	Image *VideoAsset `json:"image,omitempty"`

	// ReferenceImages, ReferenceVideos and ReferenceAudio are appended after
	// the standard input references of the same kind.
	ReferenceImages []VideoAsset `json:"referenceImages,omitempty"`
	ReferenceVideos []VideoAsset `json:"referenceVideos,omitempty"`
	ReferenceAudio  []VideoAsset `json:"referenceAudio,omitempty"`
}

// readVideoOptions reads and validates the "heygen" namespace.
func readVideoOptions(options model.ProviderOptions) (VideoOptions, error) {
	videoOptions, _, err := model.ProviderValue[VideoOptions](options, Namespace)
	if err != nil {
		return VideoOptions{}, err
	}

	if err := videoOptions.validate(); err != nil {
		return VideoOptions{}, err
	}

	return videoOptions, nil
}

func (o VideoOptions) validate() error {
	if o.Mode != nil && !validMode(*o.Mode) {
		return fmt.Errorf("%w: heygen mode %q", model.ErrInvalidArgument, *o.Mode)
	}

	if o.Resolution != nil && !validResolution(*o.Resolution) {
		return fmt.Errorf("%w: heygen resolution %q", model.ErrInvalidArgument, *o.Resolution)
	}

	if o.PromptEnhancement != nil {
		switch *o.PromptEnhancement {
		case PromptEnhancementTurbo, PromptEnhancementQuality, PromptEnhancementDisabled:
		default:
			return fmt.Errorf("%w: heygen promptEnhancement %q", model.ErrInvalidArgument, *o.PromptEnhancement)
		}
	}

	if len(o.ReferenceImages) > maxReferenceImages ||
		len(o.ReferenceVideos) > maxReferenceVideos ||
		len(o.ReferenceAudio) > maxReferenceAudio {
		return fmt.Errorf("%w: heygen accepts at most 9 reference images, 3 videos and 3 audio files", model.ErrInvalidArgument)
	}

	return o.validateAssets()
}

func (o VideoOptions) validateAssets() error {
	if o.Image != nil {
		if err := o.Image.validate(); err != nil {
			return err
		}
	}

	references := [][]VideoAsset{
		o.ReferenceImages,
		o.ReferenceVideos,
		o.ReferenceAudio,
	}

	for _, assets := range references {
		for _, asset := range assets {
			if err := asset.validate(); err != nil {
				return err
			}
		}
	}

	return nil
}

func (a VideoAsset) validate() error {
	switch a.Type {
	case AssetURL:
		if !isHTTPSURL(a.URL) {
			return fmt.Errorf("%w: heygen asset url must be an absolute HTTPS URL", model.ErrInvalidArgument)
		}

	case AssetID:
		if a.AssetID == "" {
			return fmt.Errorf("%w: heygen asset_id asset requires an assetId", model.ErrInvalidArgument)
		}

	case AssetBase64:
		if a.MediaType == "" || a.Data == "" {
			return fmt.Errorf("%w: heygen base64 asset requires a mediaType and data", model.ErrInvalidArgument)
		}

	default:
		return fmt.Errorf("%w: heygen asset type %q", model.ErrInvalidArgument, a.Type)
	}

	return nil
}

func validMode(mode VideoMode) bool {
	switch mode {
	case ModeTextToVideo, ModeImageToVideo, ModeReferenceToVideo:
		return true
	default:
		return false
	}
}

func validResolution(resolution VideoResolution) bool {
	switch resolution {
	case Resolution480p, Resolution768p, Resolution1080p, Resolution2K:
		return true
	default:
		return false
	}
}

// isHTTPSURL reports whether location is an absolute HTTPS URL.
func isHTTPSURL(location string) bool {
	if !strings.HasPrefix(location, "https://") {
		return false
	}

	parsed, err := url.Parse(location)

	return err == nil && parsed.Host != ""
}
