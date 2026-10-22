package minimax

import (
	"errors"
	"fmt"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// VideoRatio is an aspect ratio of MiniMax video models. "adaptive" follows
// the input media.
type VideoRatio string

// Video aspect ratios.
const (
	VideoRatioAdaptive VideoRatio = "adaptive"
	VideoRatio21x9     VideoRatio = "21:9"
	VideoRatio16x9     VideoRatio = "16:9"
	VideoRatio4x3      VideoRatio = "4:3"
	VideoRatio1x1      VideoRatio = "1:1"
	VideoRatio3x4      VideoRatio = "3:4"
	VideoRatio9x16     VideoRatio = "9:16"
)

// videoRatios are the accepted aspect ratios.
var videoRatios = []VideoRatio{
	VideoRatioAdaptive,
	VideoRatio21x9,
	VideoRatio16x9,
	VideoRatio4x3,
	VideoRatio1x1,
	VideoRatio3x4,
	VideoRatio9x16,
}

// VideoResolution is an output resolution tier. Availability depends on the
// model.
type VideoResolution string

// Video resolution tiers.
const (
	VideoResolution480P VideoResolution = "480P"
	VideoResolution768P VideoResolution = "768P"
	VideoResolution2K   VideoResolution = "2K"
)

// videoResolutions are the resolution tiers of every model.
var videoResolutions = []VideoResolution{
	VideoResolution480P,
	VideoResolution768P,
	VideoResolution2K,
}

// VideoOptions are the provider options of video models, read from the
// "minimax" namespace.
type VideoOptions struct {
	// Resolution is the output resolution tier.
	Resolution VideoResolution `json:"resolution,omitempty"`

	// Ratio is the aspect ratio. It overrides the aspect ratio call option.
	Ratio VideoRatio `json:"ratio,omitempty"`

	// ReferenceAudioURLs are reference audios of reference-to-video
	// generation, sent unchanged, so that they may be mm_file:// handles.
	ReferenceAudioURLs []string `json:"referenceAudioUrls,omitempty"`

	// AIGCWatermark embeds an AIGC watermark in the output.
	AIGCWatermark *bool `json:"aigcWatermark,omitempty"`

	// PollIntervalMs is the delay between status checks of Generate, 10000
	// by default.
	PollIntervalMs *int `json:"pollIntervalMs,omitempty"`

	// PollTimeoutMs is how long Generate waits, 600000 by default.
	PollTimeoutMs *int `json:"pollTimeoutMs,omitempty"`
}

func (o VideoOptions) validate() error {
	if o.Resolution != "" && !slices.Contains(videoResolutions, o.Resolution) {
		return fmt.Errorf("unsupported resolution %q", o.Resolution)
	}

	if o.Ratio != "" && !slices.Contains(videoRatios, o.Ratio) {
		return fmt.Errorf("unsupported ratio %q", o.Ratio)
	}

	if o.PollIntervalMs != nil && *o.PollIntervalMs <= 0 {
		return errors.New("pollIntervalMs must be positive")
	}

	if o.PollTimeoutMs != nil && *o.PollTimeoutMs <= 0 {
		return errors.New("pollTimeoutMs must be positive")
	}

	return nil
}

// readVideoOptions decodes and validates the video provider options.
func readVideoOptions(options model.ProviderOptions) (VideoOptions, error) {
	videoOptions, _, err := model.ProviderValue[VideoOptions](options, Namespace)
	if err != nil {
		return VideoOptions{}, err
	}

	if err := videoOptions.validate(); err != nil {
		return VideoOptions{}, fmt.Errorf("%w: minimax provider options: %w", model.ErrInvalidArgument, err)
	}

	return videoOptions, nil
}
