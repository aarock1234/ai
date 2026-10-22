package topaz

import (
	"errors"
	"fmt"
	"slices"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// videoSourceContainers are the containers Topaz accepts for the input
// video.
var videoSourceContainers = []string{
	"3gp",
	"avi",
	"dv",
	"flv",
	"m1v",
	"m2t",
	"m2ts",
	"m2v",
	"m4v",
	"mkv",
	"mov",
	"mp4",
	"mpeg",
	"mpg",
	"mts",
	"mxf",
	"ser",
	"ts",
	"vob",
	"webm",
	"wmv",
}

// videoOutputContainers are the containers Topaz can produce.
var videoOutputContainers = []string{
	"mp4",
	"mov",
	"mkv",
	"avi",
	"webm",
}

// Accepted values of the enumerated video options.
var (
	videoAudioCodecs       = []string{"AAC", "AC3", "PCM"}
	videoAudioTransfers    = []string{"Copy", "Convert", "None"}
	videoEncoders          = []string{"AV1", "H264", "H265", "ProRes", "VP9"}
	videoCompressionLevels = []string{"Low", "Mid", "High"}
	proteusVideoTypes      = []string{"Progressive", "Interlaced", "ProgressiveInterlaced"}
	proteusAutoModes       = []string{"Auto", "Manual", "Relative"}
	proteusFieldOrders     = []string{"TopFirst", "BottomFirst", "Auto"}
	proteusFocusFixLevels  = []string{"None", "Normal", "Strong"}
	proteusGrainTypes      = []string{"silver_rich", "gaussian", "grey"}
	starlightVideoCodecs   = []string{"ffv1", "prores", "vp9"}
	starlightVideoProfiles = []string{"420", "422", "444"}
)

// errAdditionalFilterModel reports an additional filter without a model.
var errAdditionalFilterModel = errors.New("additionalFilters entries must include a model key")

// VideoOptions are the provider options of video models, read from the
// "topaz" namespace.
type VideoOptions struct {
	Source *VideoSource `json:"source,omitempty"`
	Output *VideoOutput `json:"output,omitempty"`

	// AdditionalFilters are extra filters sent after the model's own, such
	// as a frame interpolation filter. Each must have a "model" key.
	AdditionalFilters []jsonobject.Object `json:"additionalFilters,omitempty"`

	// Filter holds filter settings this package does not model. It is
	// merged into the model's filter and takes precedence over the typed
	// settings.
	Filter jsonobject.Object `json:"filter,omitempty"`

	VideoFilterSettings
}

// VideoSource describes the input video. Starlight models require Width,
// Height, Duration and FrameRate, and they let other models estimate the
// cost before Topaz has the video. Media files are not inspected, so the
// values come from the caller; setting any of them requires all four.
type VideoSource struct {
	Width     *int     `json:"width,omitempty"`
	Height    *int     `json:"height,omitempty"`
	Duration  *float64 `json:"duration,omitempty"`
	FrameRate *float64 `json:"frameRate,omitempty"`

	// FrameCount is derived from Duration and FrameRate when nil, which is
	// only correct for constant frame rates.
	FrameCount *int `json:"frameCount,omitempty"`

	// Container is detected from the input media type or URL extension when
	// empty.
	Container string `json:"container,omitempty"`
}

// VideoOutput are the settings of the enhanced video.
type VideoOutput struct {
	// Width and Height take precedence over the resolution call option.
	Width  *int `json:"width,omitempty"`
	Height *int `json:"height,omitempty"`

	// FrameRate takes precedence over the fps call option. Topaz changes the
	// frame rate only with a frame interpolation filter.
	FrameRate *float64 `json:"frameRate,omitempty"`

	// AudioCodec is "AAC" (the default), "AC3" or "PCM".
	AudioCodec string `json:"audioCodec,omitempty"`

	// AudioBitrate is a bitrate such as "192k".
	AudioBitrate *string `json:"audioBitrate,omitempty"`

	// AudioTransfer is "Copy" (the default), "Convert" or "None".
	AudioTransfer string `json:"audioTransfer,omitempty"`

	// VideoEncoder is "AV1", "H264", "H265" (the Topaz default), "ProRes"
	// or "VP9". ProRes forces a mov container, AV1 and VP9 force mp4.
	VideoEncoder string `json:"videoEncoder,omitempty"`

	// VideoProfile is an encoder profile, such as "Main10" or "422 HQ".
	VideoProfile *string `json:"videoProfile,omitempty"`

	// VideoBitrate is a constant bitrate, such as "20m". VP9 requires it.
	VideoBitrate *string `json:"videoBitrate,omitempty"`

	// DynamicCompressionLevel is "Low", "Mid" or "High".
	DynamicCompressionLevel string `json:"dynamicCompressionLevel,omitempty"`

	// CropToFit center-crops to the output dimensions.
	CropToFit *bool `json:"cropToFit,omitempty"`

	// Container defaults to the input container when Topaz can produce it,
	// and to mp4 otherwise.
	Container string `json:"container,omitempty"`
}

// VideoFilterSettings are the typed settings of the model's filter, sent in
// its filters entry.
type VideoFilterSettings struct {
	// VideoType is how Proteus input frames are encoded: "Progressive",
	// "Interlaced" or "ProgressiveInterlaced".
	VideoType string `json:"videoType,omitempty"`

	// Auto is the Proteus parameter estimation: "Auto", "Manual" or
	// "Relative".
	Auto string `json:"auto,omitempty"`

	// FieldOrder is the Proteus field order of interlaced input:
	// "TopFirst", "BottomFirst" or "Auto".
	FieldOrder string `json:"fieldOrder,omitempty"`

	// FocusFixLevel is the Proteus focus fix: "None", "Normal" or "Strong".
	FocusFixLevel string `json:"focusFixLevel,omitempty"`

	// Compression, Details, Noise, Halo, Preblur and Blur are Proteus
	// adjustments from -1 to 1.
	Compression *float64 `json:"compression,omitempty"`
	Details     *float64 `json:"details,omitempty"`

	// Prenoise is the Proteus pre-processing noise reduction, 0 to 0.1.
	Prenoise *float64 `json:"prenoise,omitempty"`

	Noise   *float64 `json:"noise,omitempty"`
	Halo    *float64 `json:"halo,omitempty"`
	Preblur *float64 `json:"preblur,omitempty"`
	Blur    *float64 `json:"blur,omitempty"`

	// Grain is the Proteus grain amount, 0 to 0.1; GrainSigma 0 to 1 and
	// GrainSize 0 to 5.
	Grain      *float64 `json:"grain,omitempty"`
	GrainSigma *float64 `json:"grainSigma,omitempty"`
	GrainSize  *float64 `json:"grainSize,omitempty"`

	// GrainType is the Proteus grain model: "silver_rich", "gaussian" or
	// "grey".
	GrainType string `json:"grainType,omitempty"`

	// RecoverOriginalDetailValue is the Proteus detail recovery, 0 to 1.
	RecoverOriginalDetailValue *float64 `json:"recoverOriginalDetailValue,omitempty"`

	// Sharpness is the Starlight output sharpening, 1 to 5.
	Sharpness *float64 `json:"sharpness,omitempty"`

	// VideoBitDepth is the Starlight output bit depth.
	VideoBitDepth *int `json:"videoBitDepth,omitempty"`

	// VideoCodec is the Starlight output codec: "ffv1", "prores" or "vp9".
	VideoCodec string `json:"videoCodec,omitempty"`

	// VideoProfile is the Starlight chroma subsampling: "420", "422" or
	// "444".
	VideoProfile string `json:"videoProfile,omitempty"`

	// Watermark watermarks the Starlight output.
	Watermark *bool `json:"watermark,omitempty"`
}

// readVideoOptions decodes and validates the video provider options.
func readVideoOptions(options model.ProviderOptions) (VideoOptions, error) {
	videoOptions, _, err := model.ProviderValue[VideoOptions](options, Namespace)
	if err != nil {
		return VideoOptions{}, err
	}

	if err := videoOptions.validate(); err != nil {
		return VideoOptions{}, fmt.Errorf("%w: topaz provider options: %w", model.ErrInvalidArgument, err)
	}

	return videoOptions, nil
}

func (o VideoOptions) validate() error {
	if o.Source != nil {
		if err := o.Source.validate(); err != nil {
			return fmt.Errorf("source: %w", err)
		}
	}

	if o.Output != nil {
		if err := o.Output.validate(); err != nil {
			return fmt.Errorf("output: %w", err)
		}
	}

	for _, filter := range o.AdditionalFilters {
		if !filter.Has("model") {
			return errAdditionalFilterModel
		}
	}

	return o.VideoFilterSettings.validate()
}

// metadataIsEmpty reports whether none of the input metadata is set. The
// container is not metadata.
func (s VideoSource) metadataIsEmpty() bool {
	return s.Width == nil &&
		s.Height == nil &&
		s.Duration == nil &&
		s.FrameRate == nil &&
		s.FrameCount == nil
}

func (s VideoSource) validate() error {
	checks := []error{
		positiveInt("width", s.Width),
		positiveInt("height", s.Height),
		positiveFloat("duration", s.Duration),
		positiveFloat("frameRate", s.FrameRate),
		positiveInt("frameCount", s.FrameCount),
		oneOf("container", s.Container, videoSourceContainers),
	}

	return errors.Join(checks...)
}

func (o VideoOutput) validate() error {
	checks := []error{
		positiveInt("width", o.Width),
		positiveInt("height", o.Height),
		positiveFloat("frameRate", o.FrameRate),
		oneOf("audioCodec", o.AudioCodec, videoAudioCodecs),
		oneOf("audioTransfer", o.AudioTransfer, videoAudioTransfers),
		oneOf("videoEncoder", o.VideoEncoder, videoEncoders),
		oneOf("dynamicCompressionLevel", o.DynamicCompressionLevel, videoCompressionLevels),
		oneOf("container", o.Container, videoOutputContainers),
	}

	return errors.Join(checks...)
}

func (s VideoFilterSettings) validate() error {
	checks := []error{
		oneOf("videoType", s.VideoType, proteusVideoTypes),
		oneOf("auto", s.Auto, proteusAutoModes),
		oneOf("fieldOrder", s.FieldOrder, proteusFieldOrders),
		oneOf("focusFixLevel", s.FocusFixLevel, proteusFocusFixLevels),
		inRange("compression", s.Compression, -1, 1),
		inRange("details", s.Details, -1, 1),
		inRange("prenoise", s.Prenoise, 0, 0.1),
		inRange("noise", s.Noise, -1, 1),
		inRange("halo", s.Halo, -1, 1),
		inRange("preblur", s.Preblur, -1, 1),
		inRange("blur", s.Blur, -1, 1),
		inRange("grain", s.Grain, 0, 0.1),
		inRange("grainSigma", s.GrainSigma, 0, 1),
		inRange("grainSize", s.GrainSize, 0, 5),
		oneOf("grainType", s.GrainType, proteusGrainTypes),
		inRange("recoverOriginalDetailValue", s.RecoverOriginalDetailValue, 0, 1),
		inRange("sharpness", s.Sharpness, 1, 5),
		positiveInt("videoBitDepth", s.VideoBitDepth),
		oneOf("videoCodec", s.VideoCodec, starlightVideoCodecs),
		oneOf("videoProfile", s.VideoProfile, starlightVideoProfiles),
	}

	return errors.Join(checks...)
}

func positiveInt(name string, value *int) error {
	if value != nil && *value <= 0 {
		return fmt.Errorf("%s must be positive", name)
	}

	return nil
}

func positiveFloat(name string, value *float64) error {
	if value != nil && *value <= 0 {
		return fmt.Errorf("%s must be positive", name)
	}

	return nil
}

func inRange(name string, value *float64, minimum float64, maximum float64) error {
	if value != nil && (*value < minimum || *value > maximum) {
		return fmt.Errorf("%s must be between %g and %g", name, minimum, maximum)
	}

	return nil
}

func oneOf(name string, value string, allowed []string) error {
	if value != "" && !slices.Contains(allowed, value) {
		return fmt.Errorf("unsupported %s %q", name, value)
	}

	return nil
}
