package google

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"regexp"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/internal/origin"
	"github.com/aarock1234/ai/internal/urlpath"
	"github.com/aarock1234/ai/pkg/model"
)

// maxVideosPerCall is the largest sampleCount of Veo models.
const maxVideosPerCall = 4

// Defaults of video inputs and outputs.
const (
	defaultVideoImageMediaType = "image/png"
	generatedVideoMediaType    = "video/mp4"
	assetReferenceType         = "asset"
	cloudStorageScheme         = "gs://"
)

// videoOperationNamePattern matches the names of video operations.
var videoOperationNamePattern = regexp.MustCompile(`^models/([^/]+)/operations/([^/]+)$`)

// videoResolutions maps "WIDTHxHEIGHT" resolutions to Veo resolutions.
var videoResolutions = map[string]string{
	"1280x720":  "720p",
	"1920x1080": "1080p",
	"3840x2160": "4k",
}

// videoOptionKeys are the option keys that are not passed through as
// generation parameters.
var videoOptionKeys = map[string]bool{
	"pollIntervalMs":   true,
	"pollTimeoutMs":    true,
	"personGeneration": true,
	"negativePrompt":   true,
	"referenceImages":  true,
}

// ErrVideoGenerationFailed reports a start without an operation name or a
// completed operation without videos.
var ErrVideoGenerationFailed = errors.New("google video generation failed")

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
// "google" namespace. Other keys in the namespace are sent as generation
// parameters unchanged.
type VideoOptions struct {
	// PollIntervalMs and PollTimeoutMs are accepted for compatibility and
	// ignored; ai.VideoPollOptions controls polling.
	PollIntervalMs *float64 `json:"pollIntervalMs,omitempty"`
	PollTimeoutMs  *float64 `json:"pollTimeoutMs,omitempty"`

	PersonGeneration VideoPersonGeneration `json:"personGeneration,omitempty"`
	NegativePrompt   *string               `json:"negativePrompt,omitempty"`

	// ReferenceImages are style or asset references, used when the call has
	// no input references.
	ReferenceImages []VideoReferenceImage `json:"referenceImages,omitempty"`
}

// VideoReferenceImage is a reference image given as base64 bytes or a Cloud
// Storage URI.
type VideoReferenceImage struct {
	BytesBase64Encoded *string `json:"bytesBase64Encoded,omitempty"`
	GCSURI             *string `json:"gcsUri,omitempty"`
}

// VideoMetadata is the provider metadata of a completed generation, stored
// under the "google" namespace.
type VideoMetadata struct {
	Videos []GeneratedVideoMetadata `json:"videos"`
}

// GeneratedVideoMetadata describes one generated video.
type GeneratedVideoMetadata struct {
	// URI is the download URI without the API key.
	URI string `json:"uri"`
}

// VideoOperation identifies a long-running generation.
type VideoOperation struct {
	OperationName string `json:"operationName"`
}

// VideoModel is a Veo video generation model. Generation is asynchronous:
// Start submits a long-running operation and Status checks it.
type VideoModel struct {
	modelID  string
	provider *Provider
}

// videoRequest is the body of a predictLongRunning request.
type videoRequest struct {
	Instances  []videoInstance   `json:"instances"`
	Parameters jsonobject.Object `json:"parameters"`
}

type videoInstance struct {
	Prompt          string                `json:"prompt,omitempty"`
	Image           *videoImage           `json:"image,omitempty"`
	LastFrame       *videoImage           `json:"lastFrame,omitempty"`
	ReferenceImages []videoReferenceImage `json:"referenceImages,omitzero"`
}

type videoImage struct {
	BytesBase64Encoded string `json:"bytesBase64Encoded,omitempty"`
	GCSURI             string `json:"gcsUri,omitempty"`
	MimeType           string `json:"mimeType"`
}

type videoReferenceImage struct {
	Image         *videoImage `json:"image,omitempty"`
	ReferenceType string      `json:"referenceType,omitempty"`
}

// videoParameters are the standard generation parameters.
type videoParameters struct {
	SampleCount      int                   `json:"sampleCount"`
	AspectRatio      string                `json:"aspectRatio,omitempty"`
	Resolution       string                `json:"resolution,omitempty"`
	DurationSeconds  *float64              `json:"durationSeconds,omitempty"`
	Seed             *int                  `json:"seed,omitempty"`
	PersonGeneration VideoPersonGeneration `json:"personGeneration,omitempty"`
	NegativePrompt   *string               `json:"negativePrompt,omitempty"`
}

// videoOperationResponse is a long-running operation.
type videoOperationResponse struct {
	Name     *string               `json:"name"`
	Done     *bool                 `json:"done"`
	Error    *videoOperationError  `json:"error"`
	Response *videoOperationOutput `json:"response"`
}

// videoOperationOutput is the output of a completed operation.
type videoOperationOutput struct {
	GenerateVideoResponse *generateVideoResponse `json:"generateVideoResponse"`
}

type generateVideoResponse struct {
	GeneratedSamples []generatedVideoSample `json:"generatedSamples"`
}

type generatedVideoSample struct {
	Video *generatedVideo `json:"video"`
}

type generatedVideo struct {
	URI *string `json:"uri"`
}

// generatedSamples returns the samples of a completed operation, or nil.
func (r videoOperationResponse) generatedSamples() []generatedVideoSample {
	if r.Response == nil || r.Response.GenerateVideoResponse == nil {
		return nil
	}

	return r.Response.GenerateVideoResponse.GeneratedSamples
}

type videoOperationError struct {
	Code    *int    `json:"code"`
	Message string  `json:"message"`
	Status  *string `json:"status"`
}

// Video returns a Veo video model, such as "veo-3.1-generate-preview".
func (p *Provider) Video(modelID string) *VideoModel {
	return &VideoModel{
		modelID:  modelID,
		provider: p,
	}
}

// VideoModel returns Video(modelID).
func (p *Provider) VideoModel(modelID string) (model.VideoModel, error) {
	return p.Video(modelID), nil
}

// Provider returns the provider identifier, the provider name.
func (m *VideoModel) Provider() string {
	return m.provider.name
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

	request, err := m.provider.request(ctx, "models/"+m.modelID+":predictLongRunning", options.Headers)
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

	operationPath, err := videoOperationPath(operation.OperationName)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.request(ctx, operationPath, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.Get(ctx, request)
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

	return m.completedResult(ctx, result, status)
}

// completedResult reports the generated videos. Download URLs on the API's
// own origin carry the API key, so that they can be fetched directly.
func (m *VideoModel) completedResult(
	ctx context.Context,
	result *model.VideoStatusResult,
	status videoOperationResponse,
) (*model.VideoStatusResult, error) {
	samples := status.generatedSamples()
	if len(samples) == 0 {
		return nil, fmt.Errorf("%w: no videos in response: %s", ErrVideoGenerationFailed, result.Response.Body)
	}

	apiKey, baseURL, err := m.downloadCredentials(ctx)
	if err != nil {
		return nil, err
	}

	var metadata VideoMetadata

	for _, sample := range samples {
		if sample.Video == nil || sample.Video.URI == nil || *sample.Video.URI == "" {
			continue
		}

		uri := *sample.Video.URI

		result.Videos = append(result.Videos, model.VideoDataURL{
			URL:       authorizedVideoURL(uri, apiKey, baseURL),
			MediaType: generatedVideoMediaType,
		})
		metadata.Videos = append(metadata.Videos, GeneratedVideoMetadata{
			URI: uri,
		})
	}

	if len(result.Videos) == 0 {
		return nil, fmt.Errorf("%w: no valid videos in response", ErrVideoGenerationFailed)
	}

	result.Status = model.VideoCompleted
	result.ProviderMetadata = model.ProviderMetadata{
		Namespace: metadata,
	}

	return result, nil
}

// downloadCredentials returns the API key header value, empty when requests
// use other authentication, and the base URL.
func (m *VideoModel) downloadCredentials(ctx context.Context) (string, string, error) {
	authentication, err := m.provider.authenticationHeaders(ctx)
	if err != nil {
		return "", "", err
	}

	baseURL, err := m.provider.currentBaseURL()
	if err != nil {
		return "", "", err
	}

	headers := apicall.MergeHeaders(authentication, m.provider.headers)

	return headers.Get(apiKeyHeader), baseURL, nil
}

func (m *VideoModel) responseInfo(timestamp time.Time, response *apicall.Response) model.ResponseInfo {
	return model.ResponseInfo{
		Timestamp: timestamp,
		ModelID:   m.modelID,
		Headers:   response.Headers,
		Body:      response.Body,
	}
}

// authorizedVideoURL sets the key query parameter of a video URI on the
// base URL's origin. Other origins never receive the key.
func authorizedVideoURL(uri string, apiKey string, baseURL string) string {
	if apiKey == "" || !origin.Same(uri, baseURL) {
		return uri
	}

	parsed, err := url.Parse(uri)
	if err != nil {
		return uri
	}

	query := parsed.Query()
	query.Set("key", apiKey)
	parsed.RawQuery = query.Encode()

	return parsed.String()
}

// videoOperationPath validates an operation name, such as
// "models/veo-3/operations/abc", and escapes its segments.
func videoOperationPath(operationName string) (string, error) {
	match := videoOperationNamePattern.FindStringSubmatch(operationName)
	if match == nil {
		return "", fmt.Errorf(
			"%w: video operation: Google operation names have the form models/{model}/operations/{id}, got %q",
			model.ErrInvalidArgument,
			operationName,
		)
	}

	return "models/" + urlpath.Escape(match[1]) + "/operations/" + urlpath.Escape(match[2]), nil
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

	if references := videoInputReferences(options); references != nil {
		instance.ReferenceImages = make([]videoReferenceImage, 0, len(references))

		for _, reference := range references {
			if image := videoImageOf(reference, &warnings); image != nil {
				instance.ReferenceImages = append(instance.ReferenceImages, videoReferenceImage{
					Image:         image,
					ReferenceType: assetReferenceType,
				})
			}
		}
	} else if videoOptions.ReferenceImages != nil {
		instance.ReferenceImages = optionReferenceImages(videoOptions.ReferenceImages)
	}

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

// videoParametersObject encodes the standard parameters followed by the
// passthrough options, which replace standard parameters of the same name.
func videoParametersObject(options model.VideoOptions, videoOptions VideoOptions, extra jsonobject.Object) (jsonobject.Object, error) {
	parameters := videoParameters{
		SampleCount:      options.N,
		AspectRatio:      options.AspectRatio,
		PersonGeneration: videoOptions.PersonGeneration,
		NegativePrompt:   videoOptions.NegativePrompt,
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
// the "google" namespace.
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

	var extra jsonobject.Object
	for _, member := range raw {
		if !videoOptionKeys[member.Key] {
			extra = append(extra, member)
		}
	}

	return videoOptions, extra, nil
}

func (o VideoOptions) validate() error {
	switch o.PersonGeneration {
	case "", VideoPersonGenerationDontAllow, VideoPersonGenerationAllowAdult, VideoPersonGenerationAllowAll:
	default:
		return fmt.Errorf("%w: google personGeneration %q", model.ErrInvalidArgument, o.PersonGeneration)
	}

	if o.PollIntervalMs != nil && *o.PollIntervalMs <= 0 {
		return fmt.Errorf("%w: google pollIntervalMs must be positive", model.ErrInvalidArgument)
	}

	if o.PollTimeoutMs != nil && *o.PollTimeoutMs <= 0 {
		return fmt.Errorf("%w: google pollTimeoutMs must be positive", model.ErrInvalidArgument)
	}

	return nil
}

// optionReferenceImages converts option reference images into asset
// references. Entries with neither field are sent as empty objects.
func optionReferenceImages(images []VideoReferenceImage) []videoReferenceImage {
	references := make([]videoReferenceImage, 0, len(images))

	for _, image := range images {
		switch {
		case image.BytesBase64Encoded != nil && *image.BytesBase64Encoded != "":
			references = append(references, videoReferenceImage{
				Image: &videoImage{
					BytesBase64Encoded: *image.BytesBase64Encoded,
					MimeType:           defaultVideoImageMediaType,
				},
				ReferenceType: assetReferenceType,
			})

		case image.GCSURI != nil && *image.GCSURI != "":
			references = append(references, videoReferenceImage{
				Image: &videoImage{
					GCSURI:   *image.GCSURI,
					MimeType: defaultVideoImageMediaType,
				},
				ReferenceType: assetReferenceType,
			})

		default:
			references = append(references, videoReferenceImage{})
		}
	}

	return references
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
			"Google Generative AI video models require base64-encoded images or GCS URIs. URL will be ignored.",
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
