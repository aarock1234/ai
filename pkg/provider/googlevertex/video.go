package googlevertex

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// maxVideosPerCall is the largest sampleCount of Veo models.
const maxVideosPerCall = 4

// kebabNamespace is a further legacy metadata namespace of video results.
const kebabNamespace = "google-vertex"

// Defaults of video inputs and outputs.
const (
	defaultVideoImageMediaType = "image/png"
	generatedVideoMediaType    = "video/mp4"
	assetReferenceType         = "asset"
	cloudStorageScheme         = "gs://"
)

// videoResolutions maps "WIDTHxHEIGHT" resolutions to Veo resolutions.
var videoResolutions = map[string]string{
	"1280x720":  "720p",
	"1920x1080": "1080p",
	"3840x2160": "4k",
}

// videoOptionKeys are the option keys that are not passed through as
// generation parameters.
var videoOptionKeys = map[string]bool{
	"pollIntervalMs":     true,
	"pollTimeoutMs":      true,
	"personGeneration":   true,
	"negativePrompt":     true,
	"generateAudio":      true,
	"gcsOutputDirectory": true,
	"referenceImages":    true,
}

// videoOptionNamespaces are read in order for video options. Unlike other
// models, video models do not fall back to the "google" namespace.
var videoOptionNamespaces = []string{
	Namespace,
	LegacyNamespace,
}

// ErrVideoGenerationFailed reports a start without an operation name or a
// completed operation without videos.
var ErrVideoGenerationFailed = errors.New("google vertex video generation failed")

var (
	_ model.VideoModel    = (*VideoModel)(nil)
	_ model.VideoOperator = (*VideoModel)(nil)
	_ model.VideoProvider = (*Provider)(nil)
)

// VideoPersonGeneration controls whether generated videos may show people.
type VideoPersonGeneration string

// Person generation settings.
const (
	VideoPersonGenerationDontAllow  VideoPersonGeneration = "dont_allow"
	VideoPersonGenerationAllowAdult VideoPersonGeneration = "allow_adult"
	VideoPersonGenerationAllowAll   VideoPersonGeneration = "allow_all"
)

// VideoOptions are the provider options of video models, read from the
// "googleVertex" or "vertex" namespace. Other keys in the namespace are sent
// as generation parameters unchanged.
type VideoOptions struct {
	// PollIntervalMs and PollTimeoutMs are accepted for compatibility and
	// ignored; ai.VideoPollOptions controls polling.
	PollIntervalMs *float64 `json:"pollIntervalMs,omitempty"`
	PollTimeoutMs  *float64 `json:"pollTimeoutMs,omitempty"`

	PersonGeneration VideoPersonGeneration `json:"personGeneration,omitempty"`
	NegativePrompt   *string               `json:"negativePrompt,omitempty"`

	// GenerateAudio is used when the call does not set GenerateAudio.
	GenerateAudio *bool `json:"generateAudio,omitempty"`

	// GCSOutputDirectory is a Cloud Storage directory for the videos, which
	// are then returned as gs:// URIs.
	GCSOutputDirectory *string `json:"gcsOutputDirectory,omitempty"`

	// ReferenceImages are style or asset references, sent unchanged when the
	// call has no input references.
	ReferenceImages []VideoReferenceImage `json:"referenceImages,omitempty"`
}

// VideoReferenceImage is a reference image given as base64 bytes or a Cloud
// Storage URI.
type VideoReferenceImage struct {
	BytesBase64Encoded *string `json:"bytesBase64Encoded,omitempty"`
	GCSURI             *string `json:"gcsUri,omitempty"`
}

// VideoMetadata is the provider metadata of a completed generation, stored
// under the "googleVertex", "google-vertex" and "vertex" namespaces.
type VideoMetadata struct {
	Videos []GeneratedVideoMetadata `json:"videos"`
}

// GeneratedVideoMetadata describes one generated video.
type GeneratedVideoMetadata struct {
	GCSURI   *string `json:"gcsUri,omitempty"`
	MimeType *string `json:"mimeType,omitempty"`
}

// VideoOperation identifies a long-running generation. It is the operation
// reference of Start and Status and the body of fetchPredictOperation.
type VideoOperation struct {
	OperationName string `json:"operationName"`
}

// VideoModel is a Veo video generation model on Vertex AI. Generation is
// asynchronous: Start submits a long-running operation and Status checks it.
type VideoModel struct {
	modelID  string
	settings *settings
}

// videoRequest is the body of a predictLongRunning request.
type videoRequest struct {
	Instances  []videoInstance   `json:"instances"`
	Parameters jsonobject.Object `json:"parameters"`
}

type videoInstance struct {
	Prompt          string            `json:"prompt,omitempty"`
	Image           *videoImage       `json:"image,omitempty"`
	LastFrame       *videoImage       `json:"lastFrame,omitempty"`
	ReferenceImages []json.RawMessage `json:"referenceImages,omitzero"`
}

type videoImage struct {
	BytesBase64Encoded string `json:"bytesBase64Encoded,omitempty"`
	GCSURI             string `json:"gcsUri,omitempty"`
	MimeType           string `json:"mimeType"`
}

type videoReferenceImage struct {
	Image         videoImage `json:"image"`
	ReferenceType string     `json:"referenceType"`
}

// videoParameters are the standard generation parameters.
type videoParameters struct {
	SampleCount        int                   `json:"sampleCount"`
	AspectRatio        string                `json:"aspectRatio,omitempty"`
	Resolution         string                `json:"resolution,omitempty"`
	DurationSeconds    *float64              `json:"durationSeconds,omitempty"`
	Seed               *int                  `json:"seed,omitempty"`
	GenerateAudio      *bool                 `json:"generateAudio,omitempty"`
	PersonGeneration   VideoPersonGeneration `json:"personGeneration,omitempty"`
	NegativePrompt     *string               `json:"negativePrompt,omitempty"`
	GCSOutputDirectory *string               `json:"gcsOutputDirectory,omitempty"`
}

// videoOperationResponse is a long-running operation.
type videoOperationResponse struct {
	Name  *string `json:"name"`
	Done  *bool   `json:"done"`
	Error *struct {
		Code    *int    `json:"code"`
		Message string  `json:"message"`
		Status  *string `json:"status"`
	} `json:"error"`
	Response *struct {
		Videos []struct {
			BytesBase64Encoded *string `json:"bytesBase64Encoded"`
			GCSURI             *string `json:"gcsUri"`
			MimeType           *string `json:"mimeType"`
		} `json:"videos"`
		RAIMediaFilteredCount *int `json:"raiMediaFilteredCount"`
	} `json:"response"`
}

// Video returns a Veo video model, such as "veo-3.1-generate-001".
func (p *Provider) Video(modelID string) *VideoModel {
	return &VideoModel{
		modelID:  modelID,
		settings: &p.settings,
	}
}

// VideoModel returns Video(modelID).
func (p *Provider) VideoModel(modelID string) (model.VideoModel, error) {
	return p.Video(modelID), nil
}

// Provider returns "google.vertex.video".
func (m *VideoModel) Provider() string {
	return m.settings.name + ".video"
}

// ModelID returns the model identifier.
func (m *VideoModel) ModelID() string {
	return m.modelID
}

// MaxVideosPerCall returns 4.
func (m *VideoModel) MaxVideosPerCall() int {
	return maxVideosPerCall
}

// Start submits a generation.
func (m *VideoModel) Start(ctx context.Context, options model.VideoStartOptions) (*model.VideoStartResult, error) {
	timestamp := time.Now()

	body, warnings, err := buildVideoRequest(options.VideoOptions)
	if err != nil {
		return nil, err
	}

	request, err := m.request(ctx, ":predictLongRunning", options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, err
	}

	var operation videoOperationResponse
	if err := apicall.DecodeJSON(response.Body, &operation); err != nil {
		return nil, err
	}

	if operation.Name == nil || *operation.Name == "" {
		return nil, fmt.Errorf("%w: no operation name returned from api", ErrVideoGenerationFailed)
	}

	reference, err := json.Marshal(VideoOperation{
		OperationName: *operation.Name,
	})
	if err != nil {
		return nil, fmt.Errorf("encode video operation: %w", err)
	}

	return &model.VideoStartResult{
		Operation: reference,
		Warnings:  warnings,
		Response:  m.responseInfo(timestamp, response),
	}, nil
}

// Status checks a generation. A completed operation without videos returns
// an error wrapping ErrVideoGenerationFailed.
func (m *VideoModel) Status(ctx context.Context, options model.VideoStatusOptions) (*model.VideoStatusResult, error) {
	timestamp := time.Now()

	var operation VideoOperation
	if err := json.Unmarshal(options.Operation, &operation); err != nil {
		return nil, fmt.Errorf("%w: video operation: %w", model.ErrInvalidArgument, err)
	}

	request, err := m.request(ctx, ":fetchPredictOperation", options.Headers)
	if err != nil {
		return nil, err
	}

	// The operation reference is also the fetchPredictOperation body.
	response, err := apicall.PostJSON(ctx, request, operation)
	if err != nil {
		return nil, err
	}

	var status videoOperationResponse
	if err := apicall.DecodeJSON(response.Body, &status); err != nil {
		return nil, err
	}

	result := &model.VideoStatusResult{
		Status:   model.VideoPending,
		Response: m.responseInfo(timestamp, response),
	}

	if status.Done == nil || !*status.Done {
		return result, nil
	}

	if status.Error != nil {
		result.Status = model.VideoError
		result.Error = "Video generation failed: " + status.Error.Message

		return result, nil
	}

	return completedVideoResult(result, status)
}

// request describes a call to a method of the model's resource.
func (m *VideoModel) request(ctx context.Context, method string, headers http.Header) (apicall.Request, error) {
	baseURL, err := m.settings.resolveBaseURL(false)
	if err != nil {
		return apicall.Request{}, err
	}

	return m.settings.request(ctx, baseURL+"/models/"+m.modelID+method, headers)
}

func (m *VideoModel) responseInfo(timestamp time.Time, response *apicall.Response) model.ResponseInfo {
	return model.ResponseInfo{
		Timestamp: timestamp,
		ModelID:   m.modelID,
		Headers:   response.Headers,
		Body:      response.Body,
	}
}

// completedVideoResult reports the generated videos: inline videos are
// decoded, and Cloud Storage videos are returned as gs:// URLs.
func completedVideoResult(result *model.VideoStatusResult, status videoOperationResponse) (*model.VideoStatusResult, error) {
	if status.Response == nil || len(status.Response.Videos) == 0 {
		return nil, fmt.Errorf("%w: no videos in response: %s", ErrVideoGenerationFailed, result.Response.Body)
	}

	var metadata VideoMetadata

	for _, video := range status.Response.Videos {
		mediaType := generatedVideoMediaType
		if video.MimeType != nil && *video.MimeType != "" {
			mediaType = *video.MimeType
		}

		switch {
		case video.BytesBase64Encoded != nil && *video.BytesBase64Encoded != "":
			data, err := base64.StdEncoding.DecodeString(*video.BytesBase64Encoded)
			if err != nil {
				return nil, fmt.Errorf("%w: decode video: %w", model.ErrInvalidResponse, err)
			}

			result.Videos = append(result.Videos, model.VideoDataBytes{
				Data:      data,
				MediaType: mediaType,
			})
			metadata.Videos = append(metadata.Videos, GeneratedVideoMetadata{
				MimeType: video.MimeType,
			})

		case video.GCSURI != nil && *video.GCSURI != "":
			result.Videos = append(result.Videos, model.VideoDataURL{
				URL:       *video.GCSURI,
				MediaType: mediaType,
			})
			metadata.Videos = append(metadata.Videos, GeneratedVideoMetadata{
				GCSURI:   video.GCSURI,
				MimeType: video.MimeType,
			})
		}
	}

	if len(result.Videos) == 0 {
		return nil, fmt.Errorf("%w: no valid videos in response", ErrVideoGenerationFailed)
	}

	result.Status = model.VideoCompleted
	result.ProviderMetadata = model.ProviderMetadata{
		Namespace:       metadata,
		kebabNamespace:  metadata,
		LegacyNamespace: metadata,
	}

	return result, nil
}

// buildVideoRequest converts the call options into a predictLongRunning
// body and collects the warnings of ignored inputs.
func buildVideoRequest(options model.VideoOptions) (videoRequest, []model.Warning, error) {
	videoOptions, extra, err := readVideoOptions(options.ProviderOptions)
	if err != nil {
		return videoRequest{}, nil, err
	}

	var warnings []model.Warning

	instance := videoInstance{
		Prompt: options.Prompt,
	}

	if image := startVideoImage(options); image != nil {
		instance.Image = videoImageOf(image, &warnings)
	}

	if image := frameVideoImage(options.FrameImages, model.VideoLastFrame); image != nil {
		instance.LastFrame = videoImageOf(image, &warnings)
	}

	references, err := referenceImages(options, videoOptions, &warnings)
	if err != nil {
		return videoRequest{}, nil, err
	}

	instance.ReferenceImages = references

	parameters, err := videoParametersObject(options, videoOptions, extra)
	if err != nil {
		return videoRequest{}, nil, err
	}

	return videoRequest{
		Instances: []videoInstance{
			instance,
		},
		Parameters: parameters,
	}, warnings, nil
}

// referenceImages converts the input references into asset references, or
// passes the option reference images unchanged when there are none.
func referenceImages(
	options model.VideoOptions,
	videoOptions VideoOptions,
	warnings *[]model.Warning,
) ([]json.RawMessage, error) {
	references := videoInputReferences(options)

	if references == nil {
		return encodeEach(videoOptions.ReferenceImages)
	}

	converted := make([]videoReferenceImage, 0, len(references))
	for _, reference := range references {
		if image := videoImageOf(reference, warnings); image != nil {
			converted = append(converted, videoReferenceImage{
				Image:         *image,
				ReferenceType: assetReferenceType,
			})
		}
	}

	return encodeEach(converted)
}

// encodeEach encodes every value, keeping a nil slice nil.
func encodeEach[Value any](values []Value) ([]json.RawMessage, error) {
	if values == nil {
		return nil, nil
	}

	encoded := make([]json.RawMessage, 0, len(values))
	for _, value := range values {
		raw, err := json.Marshal(value)
		if err != nil {
			return nil, fmt.Errorf("encode reference image: %w", err)
		}

		encoded = append(encoded, raw)
	}

	return encoded, nil
}

// videoParametersObject encodes the standard parameters followed by the
// passthrough options, which replace standard parameters of the same name.
func videoParametersObject(options model.VideoOptions, videoOptions VideoOptions, extra jsonobject.Object) (jsonobject.Object, error) {
	parameters := videoParameters{
		SampleCount:        options.N,
		AspectRatio:        options.AspectRatio,
		GenerateAudio:      options.GenerateAudio,
		PersonGeneration:   videoOptions.PersonGeneration,
		NegativePrompt:     videoOptions.NegativePrompt,
		GCSOutputDirectory: videoOptions.GCSOutputDirectory,
	}

	if parameters.GenerateAudio == nil {
		parameters.GenerateAudio = videoOptions.GenerateAudio
	}

	if options.Resolution != "" {
		parameters.Resolution = options.Resolution
		if mapped, ok := videoResolutions[options.Resolution]; ok {
			parameters.Resolution = mapped
		}
	}

	if options.Duration != nil && *options.Duration != 0 {
		parameters.DurationSeconds = options.Duration
	}

	if options.Seed != nil && *options.Seed != 0 {
		parameters.Seed = options.Seed
	}

	encoded, err := json.Marshal(parameters)
	if err != nil {
		return nil, fmt.Errorf("encode video parameters: %w", err)
	}

	object, err := jsonobject.Parse(encoded)
	if err != nil {
		return nil, fmt.Errorf("encode video parameters: %w", err)
	}

	return object.Merge(extra), nil
}

// readVideoOptions reads the typed options and the passthrough members of
// the first video options namespace that is set.
func readVideoOptions(options model.ProviderOptions) (VideoOptions, jsonobject.Object, error) {
	for _, namespace := range videoOptionNamespaces {
		raw, ok, err := model.ProviderValue[jsonobject.Object](options, namespace)
		if err != nil {
			return VideoOptions{}, nil, err
		}

		if !ok {
			continue
		}

		videoOptions, _, err := model.ProviderValue[VideoOptions](options, namespace)
		if err != nil {
			return VideoOptions{}, nil, err
		}

		if err := videoOptions.validate(); err != nil {
			return VideoOptions{}, nil, err
		}

		return videoOptions, passthroughMembers(raw), nil
	}

	return VideoOptions{}, nil, nil
}

func passthroughMembers(raw jsonobject.Object) jsonobject.Object {
	var extra jsonobject.Object

	for _, member := range raw {
		if !videoOptionKeys[member.Key] {
			extra = append(extra, member)
		}
	}

	return extra
}

func (o VideoOptions) validate() error {
	switch o.PersonGeneration {
	case "", VideoPersonGenerationDontAllow, VideoPersonGenerationAllowAdult, VideoPersonGenerationAllowAll:
	default:
		return fmt.Errorf("%w: google vertex personGeneration %q", model.ErrInvalidArgument, o.PersonGeneration)
	}

	if o.PollIntervalMs != nil && *o.PollIntervalMs <= 0 {
		return fmt.Errorf("%w: google vertex pollIntervalMs must be positive", model.ErrInvalidArgument)
	}

	if o.PollTimeoutMs != nil && *o.PollTimeoutMs <= 0 {
		return fmt.Errorf("%w: google vertex pollTimeoutMs must be positive", model.ErrInvalidArgument)
	}

	return nil
}

// videoImageOf converts an input file. Veo accepts inline bytes and Cloud
// Storage URIs; other URLs are ignored with a warning.
func videoImageOf(file model.VideoFile, warnings *[]model.Warning) *videoImage {
	switch typed := file.(type) {
	case model.VideoURL:
		if strings.HasPrefix(typed.URL, cloudStorageScheme) {
			return &videoImage{
				GCSURI:   typed.URL,
				MimeType: defaultVideoImageMediaType,
			}
		}

		*warnings = append(*warnings, model.UnsupportedWarning(
			"URL-based image input",
			"Vertex AI video models require base64-encoded images or GCS URIs. URL will be ignored.",
		))

		return nil

	case model.VideoBytes:
		mediaType := typed.MediaType
		if mediaType == "" {
			mediaType = defaultVideoImageMediaType
		}

		return &videoImage{
			BytesBase64Encoded: base64.StdEncoding.EncodeToString(typed.Data),
			MimeType:           mediaType,
		}

	default:
		return nil
	}
}

// startVideoImage is the first frame image, or the image option.
func startVideoImage(options model.VideoOptions) model.VideoFile {
	if first := frameVideoImage(options.FrameImages, model.VideoFirstFrame); first != nil {
		return first
	}

	return options.Image
}

func frameVideoImage(frames []model.VideoFrameImage, frameType model.VideoFrameType) model.VideoFile {
	for _, frame := range frames {
		if frame.FrameType == frameType {
			return frame.Image
		}
	}

	return nil
}

// videoInputReferences returns the input references, or nil when there are
// none or frame images exclude them.
func videoInputReferences(options model.VideoOptions) []model.VideoFile {
	if len(options.FrameImages) > 0 || len(options.InputReferences) == 0 {
		return nil
	}

	return options.InputReferences
}
