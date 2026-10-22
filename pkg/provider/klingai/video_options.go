package klingai

import (
	"fmt"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// VideoMode is the generation quality mode.
type VideoMode string

// Generation modes.
const (
	// VideoModeStandard is cost-effective.
	VideoModeStandard VideoMode = "std"

	// VideoModePro has higher quality and takes longer.
	VideoModePro VideoMode = "pro"
)

// Sound controls audio generation.
type Sound string

// Sound settings.
const (
	SoundOn  Sound = "on"
	SoundOff Sound = "off"
)

// CameraControlType is a camera movement.
type CameraControlType string

// Camera movements.
const (
	CameraSimple           CameraControlType = "simple"
	CameraDownBack         CameraControlType = "down_back"
	CameraForwardUp        CameraControlType = "forward_up"
	CameraRightTurnForward CameraControlType = "right_turn_forward"
	CameraLeftTurnForward  CameraControlType = "left_turn_forward"
)

// ShotType is the storyboard method of multi-shot generation.
type ShotType string

// Storyboard methods.
const (
	// ShotTypeCustomize uses the shots of MultiPrompt.
	ShotTypeCustomize ShotType = "customize"

	// ShotTypeIntelligence segments the main prompt automatically.
	ShotTypeIntelligence ShotType = "intelligence"
)

// CharacterOrientation is the orientation of motion control characters.
type CharacterOrientation string

// Character orientations.
const (
	// OrientationImage follows the person in the image. The reference video
	// may be up to 10 seconds long.
	OrientationImage CharacterOrientation = "image"

	// OrientationVideo follows the person in the video. The reference video
	// may be up to 30 seconds long.
	OrientationVideo CharacterOrientation = "video"
)

// KeepOriginalSound controls the sound of the motion control reference
// video.
type KeepOriginalSound string

// Original sound settings.
const (
	KeepOriginalSoundYes KeepOriginalSound = "yes"
	KeepOriginalSoundNo  KeepOriginalSound = "no"
)

// videoOptionKeys are the VideoOptions members, which are mapped to request
// fields rather than passed through.
var videoOptionKeys = map[string]bool{
	"mode":                 true,
	"pollIntervalMs":       true,
	"pollTimeoutMs":        true,
	"negativePrompt":       true,
	"sound":                true,
	"cfgScale":             true,
	"cameraControl":        true,
	"multiShot":            true,
	"shotType":             true,
	"multiPrompt":          true,
	"elementList":          true,
	"voiceList":            true,
	"imageTail":            true,
	"staticMask":           true,
	"dynamicMasks":         true,
	"videoUrl":             true,
	"characterOrientation": true,
	"keepOriginalSound":    true,
	"watermarkEnabled":     true,
}

// VideoOptions are the provider options of video models, read from the
// "klingai" namespace. Other members of the namespace are sent in the
// request body unchanged. Not every option is supported by every model
// version and mode.
type VideoOptions struct {
	// Mode is the quality mode. Motion control requires it.
	Mode *VideoMode `json:"mode,omitempty"`

	// PollIntervalMs and PollTimeoutMs are accepted for compatibility and
	// ignored; ai.VideoPollOptions controls polling.
	PollIntervalMs *float64 `json:"pollIntervalMs,omitempty"`
	PollTimeoutMs  *float64 `json:"pollTimeoutMs,omitempty"`

	// NegativePrompt describes what to avoid, in up to 2500 characters.
	NegativePrompt *string `json:"negativePrompt,omitempty"`

	// Sound generates audio, on v2.6 and later in pro mode. The standard
	// GenerateAudio option takes precedence.
	Sound *Sound `json:"sound,omitempty"`

	// CFGScale is the prompt relevance, from 0 to 1. Kling v2.x models do
	// not support it.
	CFGScale *float64 `json:"cfgScale,omitempty"`

	// CameraControl selects the camera movement. Without it, the model
	// chooses one.
	CameraControl *CameraControl `json:"cameraControl,omitempty"`

	// ImageTail is the image-to-video end frame, as a URL or raw base64. A
	// last-frame image takes precedence.
	ImageTail *string `json:"imageTail,omitempty"`

	// StaticMask is the image-to-video static brush mask, as a URL or raw
	// base64.
	StaticMask *string `json:"staticMask,omitempty"`

	// DynamicMasks are up to 6 image-to-video motion brush groups.
	DynamicMasks []DynamicMask `json:"dynamicMasks,omitempty"`

	// MultiShot splits a v3.0 video into up to 6 storyboard shots.
	MultiShot *bool `json:"multiShot,omitempty"`

	// ShotType is the storyboard method of multi-shot generation.
	ShotType *ShotType `json:"shotType,omitempty"`

	// MultiPrompt are the shots of customized multi-shot generation.
	MultiPrompt []MultiPromptShot `json:"multiPrompt,omitempty"`

	// ElementList are v3.0 reference elements of image-to-video and motion
	// control.
	ElementList []Element `json:"elementList,omitempty"`

	// VoiceList are up to 2 v3.0 voice references, used in the prompt as
	// <<<voice_1>>>.
	VoiceList []Voice `json:"voiceList,omitempty"`

	// WatermarkEnabled also generates watermarked results.
	WatermarkEnabled *bool `json:"watermarkEnabled,omitempty"`

	// VideoURL is the motion control reference video, whose character
	// actions the generated video follows.
	VideoURL *string `json:"videoUrl,omitempty"`

	// CharacterOrientation is the motion control character orientation.
	CharacterOrientation *CharacterOrientation `json:"characterOrientation,omitempty"`

	// KeepOriginalSound keeps the sound of the motion control reference
	// video. The default is yes.
	KeepOriginalSound *KeepOriginalSound `json:"keepOriginalSound,omitempty"`
}

// CameraControl is a camera movement and its configuration.
type CameraControl struct {
	Type   CameraControlType `json:"type"`
	Config *CameraConfig     `json:"config,omitempty"`
}

// CameraConfig configures a simple camera movement.
type CameraConfig struct {
	Horizontal *float64 `json:"horizontal,omitempty"`
	Vertical   *float64 `json:"vertical,omitempty"`
	Pan        *float64 `json:"pan,omitempty"`
	Tilt       *float64 `json:"tilt,omitempty"`
	Roll       *float64 `json:"roll,omitempty"`
	Zoom       *float64 `json:"zoom,omitempty"`
}

// DynamicMask is a motion brush mask and its trajectory.
type DynamicMask struct {
	Mask         string       `json:"mask"`
	Trajectories []Trajectory `json:"trajectories"`
}

// Trajectory is one point of a motion brush trajectory.
type Trajectory struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}

// MultiPromptShot is one storyboard shot. The shot durations must add up to
// the video duration.
type MultiPromptShot struct {
	Index float64 `json:"index"`

	// Prompt describes the shot in up to 512 characters.
	Prompt string `json:"prompt"`

	// Duration is the shot length in seconds.
	Duration string `json:"duration"`
}

// Element references a KlingAI element.
type Element struct {
	ElementID float64 `json:"element_id"`
}

// Voice references a KlingAI voice.
type Voice struct {
	VoiceID string `json:"voice_id"`
}

// readVideoOptions reads the typed options and the passthrough members of
// the "klingai" namespace.
func readVideoOptions(options model.ProviderOptions) (VideoOptions, jsonobject.Object, error) {
	videoOptions, _, err := model.ProviderValue[VideoOptions](options, Namespace)
	if err != nil {
		return VideoOptions{}, nil, err
	}

	if err := videoOptions.validate(); err != nil {
		return VideoOptions{}, nil, err
	}

	raw, _, err := model.ProviderValue[jsonobject.Object](options, Namespace)
	if err != nil {
		return VideoOptions{}, nil, err
	}

	var passthrough jsonobject.Object
	for _, member := range raw {
		if !videoOptionKeys[member.Key] {
			passthrough = append(passthrough, member)
		}
	}

	return videoOptions, passthrough, nil
}

func (o VideoOptions) validate() error {
	if err := o.validateEnums(); err != nil {
		return err
	}

	if o.PollIntervalMs != nil && *o.PollIntervalMs <= 0 {
		return fmt.Errorf("%w: klingai pollIntervalMs must be positive", model.ErrInvalidArgument)
	}

	if o.PollTimeoutMs != nil && *o.PollTimeoutMs <= 0 {
		return fmt.Errorf("%w: klingai pollTimeoutMs must be positive", model.ErrInvalidArgument)
	}

	return nil
}

func (o VideoOptions) validateEnums() error {
	if o.Mode != nil && *o.Mode != VideoModeStandard && *o.Mode != VideoModePro {
		return fmt.Errorf("%w: klingai mode %q", model.ErrInvalidArgument, *o.Mode)
	}

	if o.Sound != nil && *o.Sound != SoundOn && *o.Sound != SoundOff {
		return fmt.Errorf("%w: klingai sound %q", model.ErrInvalidArgument, *o.Sound)
	}

	if o.CameraControl != nil {
		switch o.CameraControl.Type {
		case CameraSimple, CameraDownBack, CameraForwardUp, CameraRightTurnForward, CameraLeftTurnForward:
		default:
			return fmt.Errorf("%w: klingai cameraControl type %q", model.ErrInvalidArgument, o.CameraControl.Type)
		}
	}

	if o.ShotType != nil && *o.ShotType != ShotTypeCustomize && *o.ShotType != ShotTypeIntelligence {
		return fmt.Errorf("%w: klingai shotType %q", model.ErrInvalidArgument, *o.ShotType)
	}

	if o.CharacterOrientation != nil && *o.CharacterOrientation != OrientationImage && *o.CharacterOrientation != OrientationVideo {
		return fmt.Errorf("%w: klingai characterOrientation %q", model.ErrInvalidArgument, *o.CharacterOrientation)
	}

	if o.KeepOriginalSound != nil && *o.KeepOriginalSound != KeepOriginalSoundYes && *o.KeepOriginalSound != KeepOriginalSoundNo {
		return fmt.Errorf("%w: klingai keepOriginalSound %q", model.ErrInvalidArgument, *o.KeepOriginalSound)
	}

	return nil
}
