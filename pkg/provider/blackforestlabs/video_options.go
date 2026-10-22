package blackforestlabs

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// Option limits of FLUX 3 video.
const (
	maxVideoKeyframes       = 10
	maxKeyframeTimestamp    = 20
	maxVideoSafetyTolerance = 4
	latestVideoVersion      = "latest"
)

// VideoResolution is an output resolution tier of FLUX 3 video.
type VideoResolution string

// Video resolution tiers.
const (
	VideoResolutionHD  VideoResolution = "hd"
	VideoResolutionFHD VideoResolution = "fhd"
)

// videoResolutions are the accepted resolution tiers.
var videoResolutions = []VideoResolution{
	VideoResolutionHD,
	VideoResolutionFHD,
}

// videoAspectRatios are the aspect ratios FLUX 3 video accepts. "auto" lets
// the model infer the ratio from the prompt and conditioning media, and is
// the API default.
var videoAspectRatios = []string{
	"21:9",
	"2:1",
	"16:9",
	"4:3",
	"1:1",
	"3:4",
	"9:16",
	"auto",
}

// errMixedKeyframes reports keyframes that combine timed and untimed
// entries.
var errMixedKeyframes = errors.New("keyframes must be all images or all timed images")

// VideoOptions are the provider options of video models, read from the
// "blackForestLabs" namespace.
type VideoOptions struct {
	// Resolution is the output resolution tier. It takes precedence over
	// the resolution call option, which is mapped onto a tier.
	Resolution VideoResolution `json:"resolution,omitempty"`

	// AspectRatio is the aspect ratio of the video. It takes precedence over
	// the aspect ratio call option and, unlike it, can be "auto".
	AspectRatio string `json:"aspectRatio,omitempty"`

	// Keyframes are the images of image-to-video generation, for shapes the
	// image and frame image call options cannot express: more than two
	// images, or images pinned to a second. They take precedence over both.
	// Three or more untimed images require a duration.
	Keyframes VideoKeyframes `json:"keyframes,omitempty"`

	// SafetyTolerance is the moderation strictness from 0 (strictest) to 4.
	SafetyTolerance *int `json:"safetyTolerance,omitempty"`

	// Draft renders a fast, lower-quality preview.
	Draft *bool `json:"draft,omitempty"`

	// DraftCache is the encrypted bundle of a prior draft generation, as
	// base64 or its URL. It switches the request to draft enhance, which
	// replays the bundle at full quality.
	DraftCache *string `json:"draftCache,omitempty"`

	// Version pins the model version. Only "latest" is available.
	Version *string `json:"version,omitempty"`
}

// VideoKeyframe is one keyframe: an image URL or base64 image, optionally
// pinned to a second between 0 and 20.
type VideoKeyframe struct {
	Image string

	// Timestamp is nil for keyframes the API spaces evenly.
	Timestamp *float64
}

// VideoKeyframes are the keyframes of a generation. In JSON they are an
// array of image strings or an array of [seconds, image] pairs.
type VideoKeyframes []VideoKeyframe

// UnmarshalJSON accepts an array of strings or of [seconds, image] pairs.
func (k *VideoKeyframes) UnmarshalJSON(data []byte) error {
	var entries []json.RawMessage
	if err := json.Unmarshal(data, &entries); err != nil {
		return err
	}

	keyframes := make(VideoKeyframes, 0, len(entries))

	for _, entry := range entries {
		keyframe, err := decodeKeyframe(entry)
		if err != nil {
			return err
		}

		keyframes = append(keyframes, keyframe)
	}

	*k = keyframes

	return nil
}

// MarshalJSON encodes untimed keyframes as strings and timed keyframes as
// [seconds, image] pairs.
func (k VideoKeyframes) MarshalJSON() ([]byte, error) {
	entries := make([]any, 0, len(k))

	for _, keyframe := range k {
		if keyframe.Timestamp == nil {
			entries = append(entries, keyframe.Image)

			continue
		}

		entries = append(entries, []any{
			*keyframe.Timestamp,
			keyframe.Image,
		})
	}

	return json.Marshal(entries)
}

func decodeKeyframe(entry json.RawMessage) (VideoKeyframe, error) {
	if bytes.HasPrefix(bytes.TrimSpace(entry), []byte(`"`)) {
		var image string
		if err := json.Unmarshal(entry, &image); err != nil {
			return VideoKeyframe{}, err
		}

		return VideoKeyframe{
			Image: image,
		}, nil
	}

	var pair []json.RawMessage
	if err := json.Unmarshal(entry, &pair); err != nil {
		return VideoKeyframe{}, err
	}

	if len(pair) != 2 {
		return VideoKeyframe{}, errors.New("timed keyframe must be a [seconds, image] pair")
	}

	var timestamp float64
	if err := json.Unmarshal(pair[0], &timestamp); err != nil {
		return VideoKeyframe{}, err
	}

	var image string
	if err := json.Unmarshal(pair[1], &image); err != nil {
		return VideoKeyframe{}, err
	}

	return VideoKeyframe{
		Image:     image,
		Timestamp: &timestamp,
	}, nil
}

// untimed returns the number of keyframes without a timestamp.
func (k VideoKeyframes) untimed() int {
	count := 0

	for _, keyframe := range k {
		if keyframe.Timestamp == nil {
			count++
		}
	}

	return count
}

func (k VideoKeyframes) validate() error {
	if k == nil {
		return nil
	}

	if len(k) == 0 || len(k) > maxVideoKeyframes {
		return fmt.Errorf("keyframes must have between 1 and %d entries", maxVideoKeyframes)
	}

	untimed := k.untimed()
	if untimed == len(k) {
		return nil
	}

	if untimed > 0 {
		return errMixedKeyframes
	}

	for index, keyframe := range k {
		if *keyframe.Timestamp < 0 || *keyframe.Timestamp > maxKeyframeTimestamp {
			return fmt.Errorf("keyframe timestamps must be between 0 and %d", maxKeyframeTimestamp)
		}

		if index > 0 && *keyframe.Timestamp <= *k[index-1].Timestamp {
			return errors.New("timed keyframes must be in chronological order")
		}
	}

	return nil
}

func (o VideoOptions) validate() error {
	if o.Resolution != "" && !slices.Contains(videoResolutions, o.Resolution) {
		return fmt.Errorf("resolution %q is not one of hd or fhd", o.Resolution)
	}

	if o.AspectRatio != "" && !slices.Contains(videoAspectRatios, o.AspectRatio) {
		return fmt.Errorf("unsupported aspectRatio %q", o.AspectRatio)
	}

	if err := o.Keyframes.validate(); err != nil {
		return err
	}

	if o.SafetyTolerance != nil && (*o.SafetyTolerance < 0 || *o.SafetyTolerance > maxVideoSafetyTolerance) {
		return fmt.Errorf("safetyTolerance must be between 0 and %d", maxVideoSafetyTolerance)
	}

	if o.Version != nil && *o.Version != latestVideoVersion {
		return fmt.Errorf("version must be %q", latestVideoVersion)
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
		return VideoOptions{}, fmt.Errorf("%w: blackForestLabs provider options: %w", model.ErrInvalidArgument, err)
	}

	return videoOptions, nil
}
