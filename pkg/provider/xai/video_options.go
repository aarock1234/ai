package xai

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"slices"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// Limits of the video options.
const (
	maxReferenceImageURLs    = 7
	maxReferenceVoiceIDs     = 3
	maxVideoKeyframes        = 4
	maxStorageExpiresAfter   = 2592000
	minPublicURLExpiresAfter = 3600
)

// VideoMode selects the video operation.
type VideoMode string

// Video modes. Without a mode, a request generates from text or an image.
const (
	// VideoModeEditVideo edits VideoOptions.VideoURL.
	VideoModeEditVideo VideoMode = "edit-video"

	// VideoModeExtendVideo extends VideoOptions.VideoURL from its last
	// frame.
	VideoModeExtendVideo VideoMode = "extend-video"

	// VideoModeReferenceToVideo generates from reference images and audio.
	VideoModeReferenceToVideo VideoMode = "reference-to-video"
)

// VideoResolution is a resolution of xAI video models.
type VideoResolution string

// Video resolutions. 1080p requires grok-imagine-video-1.5.
const (
	VideoResolution480p  VideoResolution = "480p"
	VideoResolution720p  VideoResolution = "720p"
	VideoResolution1080p VideoResolution = "1080p"
)

// videoModes and videoResolutionValues are the accepted values.
var (
	videoModes = []VideoMode{
		VideoModeEditVideo,
		VideoModeExtendVideo,
		VideoModeReferenceToVideo,
	}

	videoResolutionValues = []VideoResolution{
		VideoResolution480p,
		VideoResolution720p,
		VideoResolution1080p,
	}
)

// videoOptionKeys are the option keys that are not passed through to the
// request body.
var videoOptionKeys = []string{
	"mode",
	"pollIntervalMs",
	"pollTimeoutMs",
	"resolution",
	"videoUrl",
	"referenceImageUrls",
	"referenceVoiceIds",
	"keyframes",
	"storageOptions",
	"user",
}

// VideoOptions are the provider options of video models, read from the
// "xai" namespace. Other keys in the namespace are sent in the request body
// unchanged.
//
// Without Mode, a VideoURL selects edit-video, and image or audio input
// references or ReferenceImageURLs select reference-to-video.
type VideoOptions struct {
	Mode VideoMode `json:"mode,omitempty"`

	// VideoURL is the source video of edit-video and extend-video.
	VideoURL *string `json:"videoUrl,omitempty"`

	// ReferenceImageURLs are 1 to 7 reference images of reference-to-video
	// generation, used when the call has no image input references.
	ReferenceImageURLs []string `json:"referenceImageUrls,omitempty"`

	// ReferenceVoiceIDs are up to 3 preset voices that give the subject of
	// reference-to-video generation a voice.
	ReferenceVoiceIDs []string `json:"referenceVoiceIds,omitempty"`

	// Keyframes are up to 4 mid-video image anchors of
	// grok-imagine-video-1.5 generation.
	Keyframes []VideoKeyframe `json:"keyframes,omitempty"`

	// StorageOptions store the video in the xAI Files API.
	StorageOptions *VideoStorageOptions `json:"storageOptions,omitempty"`

	// User identifies the end user for abuse monitoring. Video extension
	// does not send it.
	User *string `json:"user,omitempty"`

	// PollIntervalMs and PollTimeoutMs are accepted for compatibility and
	// ignored; ai.VideoPollOptions controls polling.
	PollIntervalMs *float64 `json:"pollIntervalMs,omitempty"`
	PollTimeoutMs  *float64 `json:"pollTimeoutMs,omitempty"`

	// Resolution takes precedence over the resolution call option.
	Resolution VideoResolution `json:"resolution,omitempty"`
}

// VideoKeyframe anchors an image at a time of the video.
type VideoKeyframe struct {
	ImageURL         string  `json:"imageUrl"`
	TimestampSeconds float64 `json:"timestampSeconds"`
}

// VideoStorageOptions store a generated video in the Files API.
type VideoStorageOptions struct {
	Filename string `json:"filename"`

	// ExpiresAfter is the lifetime of the stored file in seconds, at most
	// 2592000.
	ExpiresAfter *int `json:"expiresAfter,omitempty"`

	// PublicURL creates a public URL for the stored file.
	PublicURL *VideoPublicURL `json:"publicUrl,omitempty"`
}

// VideoPublicURL requests a public URL: true or false, or an object with
// the URL's lifetime. In JSON it is a boolean or {"expiresAfter": seconds}.
type VideoPublicURL struct {
	// Enabled is set when the option is a boolean.
	Enabled *bool

	// ExpiresAfter is the lifetime of the URL in seconds, between 3600 and
	// 2592000, when the option is an object.
	ExpiresAfter *int
}

type publicURLObject struct {
	ExpiresAfter *int `json:"expiresAfter,omitempty"`
}

// UnmarshalJSON accepts a boolean or an object.
func (u *VideoPublicURL) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)

	if bytes.Equal(trimmed, []byte("true")) || bytes.Equal(trimmed, []byte("false")) {
		var enabled bool
		if err := json.Unmarshal(trimmed, &enabled); err != nil {
			return err
		}

		*u = VideoPublicURL{
			Enabled: &enabled,
		}

		return nil
	}

	var object publicURLObject
	if err := json.Unmarshal(trimmed, &object); err != nil {
		return err
	}

	*u = VideoPublicURL{
		ExpiresAfter: object.ExpiresAfter,
	}

	return nil
}

// MarshalJSON encodes a boolean, or an object when Enabled is nil.
func (u VideoPublicURL) MarshalJSON() ([]byte, error) {
	if u.Enabled != nil {
		return json.Marshal(*u.Enabled)
	}

	return json.Marshal(publicURLObject{
		ExpiresAfter: u.ExpiresAfter,
	})
}

func (o VideoOptions) validate() error {
	if o.Mode != "" && !slices.Contains(videoModes, o.Mode) {
		return fmt.Errorf("unsupported mode %q", o.Mode)
	}

	if o.VideoURL != nil && *o.VideoURL == "" {
		return errors.New("videoUrl must not be empty")
	}

	if o.ReferenceImageURLs != nil && (len(o.ReferenceImageURLs) == 0 || len(o.ReferenceImageURLs) > maxReferenceImageURLs) {
		return fmt.Errorf("referenceImageUrls must have between 1 and %d entries", maxReferenceImageURLs)
	}

	if slices.Contains(o.ReferenceImageURLs, "") {
		return errors.New("referenceImageUrls entries must not be empty")
	}

	if slices.Contains(o.ReferenceVoiceIDs, "") {
		return errors.New("referenceVoiceIds entries must not be empty")
	}

	if len(o.ReferenceVoiceIDs) > maxReferenceVoiceIDs {
		return fmt.Errorf("referenceVoiceIds must have at most %d entries", maxReferenceVoiceIDs)
	}

	if err := validateKeyframes(o.Keyframes); err != nil {
		return err
	}

	if o.StorageOptions != nil {
		if err := o.StorageOptions.validate(); err != nil {
			return err
		}
	}

	if o.PollIntervalMs != nil && *o.PollIntervalMs <= 0 {
		return errors.New("pollIntervalMs must be positive")
	}

	if o.PollTimeoutMs != nil && *o.PollTimeoutMs <= 0 {
		return errors.New("pollTimeoutMs must be positive")
	}

	if o.Resolution != "" && !slices.Contains(videoResolutionValues, o.Resolution) {
		return fmt.Errorf("unsupported resolution %q", o.Resolution)
	}

	return nil
}

func validateKeyframes(keyframes []VideoKeyframe) error {
	if len(keyframes) > maxVideoKeyframes {
		return fmt.Errorf("keyframes must have at most %d entries", maxVideoKeyframes)
	}

	for _, keyframe := range keyframes {
		if keyframe.ImageURL == "" {
			return errors.New("keyframe imageUrl must not be empty")
		}

		if keyframe.TimestampSeconds <= 0 {
			return errors.New("keyframe timestampSeconds must be positive")
		}
	}

	return nil
}

func (s VideoStorageOptions) validate() error {
	if s.Filename == "" {
		return errors.New("storageOptions.filename must not be empty")
	}

	if s.ExpiresAfter != nil && (*s.ExpiresAfter <= 0 || *s.ExpiresAfter > maxStorageExpiresAfter) {
		return fmt.Errorf("storageOptions.expiresAfter must be between 1 and %d", maxStorageExpiresAfter)
	}

	publicURL := s.PublicURL
	if publicURL != nil && publicURL.ExpiresAfter != nil {
		expiresAfter := *publicURL.ExpiresAfter
		if expiresAfter < minPublicURLExpiresAfter || expiresAfter > maxStorageExpiresAfter {
			return fmt.Errorf(
				"storageOptions.publicUrl.expiresAfter must be between %d and %d",
				minPublicURLExpiresAfter,
				maxStorageExpiresAfter,
			)
		}
	}

	return nil
}

// readVideoOptions reads the typed options and the passthrough members of
// the "xai" namespace.
func readVideoOptions(options model.ProviderOptions) (VideoOptions, jsonobject.Object, error) {
	videoOptions, _, err := model.ProviderValue[VideoOptions](options, Namespace)
	if err != nil {
		return VideoOptions{}, nil, err
	}

	if err := videoOptions.validate(); err != nil {
		return VideoOptions{}, nil, fmt.Errorf("%w: xai provider options: %w", model.ErrInvalidArgument, err)
	}

	raw, _, err := model.ProviderValue[jsonobject.Object](options, Namespace)
	if err != nil {
		return VideoOptions{}, nil, err
	}

	var extra jsonobject.Object
	for _, member := range raw {
		if !slices.Contains(videoOptionKeys, member.Key) {
			extra = append(extra, member)
		}
	}

	return videoOptions, extra, nil
}
