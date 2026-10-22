package prodia

import (
	"context"
	"encoding/json"
	"fmt"
	"mime"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/download"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// videoAccept asks for the job record and the video as multipart parts.
	videoAccept = "multipart/form-data; video/mp4"

	// defaultVideoMediaType is the media type of output parts without a
	// video content type.
	defaultVideoMediaType = "video/mp4"

	// unknownMediaType is the media type of downloaded inputs without one.
	unknownMediaType = "application/octet-stream"
)

// inputExtensions are the file extensions of video job inputs.
var inputExtensions = map[string]string{
	"image/png":  ".png",
	"image/jpeg": ".jpg",
	"image/webp": ".webp",
	"video/mp4":  ".mp4",
	"video/webm": ".webm",
}

var (
	_ model.VideoModel     = (*VideoModel)(nil)
	_ model.VideoGenerator = (*VideoModel)(nil)
	_ model.VideoProvider  = (*Provider)(nil)
)

// VideoOptions are the provider options of video models, read from the
// "prodia" namespace.
type VideoOptions struct {
	// Resolution is the video resolution, such as "480p" or "720p".
	Resolution *string `json:"resolution,omitempty"`
}

// VideoMetadata is the Prodia metadata of a video response, stored under
// ProviderMetadata["prodia"].
type VideoMetadata struct {
	// Videos has one entry for the generated video.
	Videos []JobMetadata `json:"videos"`
}

// VideoModel is a Prodia video model. It generates one video per call:
// text-to-video, or image-to-video when the call has an image.
type VideoModel struct {
	modelID  string
	provider *Provider
}

type videoJobRequest struct {
	Type   string         `json:"type"`
	Config videoJobConfig `json:"config"`
}

type videoJobConfig struct {
	Prompt     string  `json:"prompt,omitempty"`
	Seed       *int    `json:"seed,omitempty"`
	Resolution *string `json:"resolution,omitempty"`
}

// inputFile is the image of an image-to-video job.
type inputFile struct {
	data      []byte
	mediaType string
}

// Video returns a video model, such as
// "inference.wan2-2.lightning.txt2vid.v0".
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

// Provider returns the provider identifier, "prodia.video" by default.
func (m *VideoModel) Provider() string {
	return m.provider.name + ".video"
}

// ModelID returns the model identifier.
func (m *VideoModel) ModelID() string {
	return m.modelID
}

// MaxVideosPerCall returns 1.
func (m *VideoModel) MaxVideosPerCall() int {
	return 1
}

// Generate runs a video job and returns its video. A call with an image
// sends it with the job as multipart form data.
func (m *VideoModel) Generate(ctx context.Context, options model.VideoOptions) (*model.VideoResult, error) {
	timestamp := time.Now()

	videoOptions, _, err := model.ProviderValue[VideoOptions](options.ProviderOptions, Namespace)
	if err != nil {
		return nil, err
	}

	job := videoJobRequest{
		Type: m.modelID,
		Config: videoJobConfig{
			Prompt:     options.Prompt,
			Seed:       options.Seed,
			Resolution: videoOptions.Resolution,
		},
	}

	response, err := m.submitJob(ctx, job, options)
	if err != nil {
		return nil, err
	}

	record, video, err := videoJobResult(response)
	if err != nil {
		return nil, err
	}

	return &model.VideoResult{
		Videos: []model.VideoData{
			video,
		},
		ProviderMetadata: model.ProviderMetadata{
			Namespace: VideoMetadata{
				Videos: []JobMetadata{
					jobMetadata(record),
				},
			},
		},
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: timestamp,
				ModelID:   m.modelID,
			},
			Headers: response.Headers,
		},
	}, nil
}

// submitJob sends a text-to-video job as JSON, or an image-to-video job as
// multipart form data with its input image.
func (m *VideoModel) submitJob(ctx context.Context, job videoJobRequest, options model.VideoOptions) (*apicall.Response, error) {
	request, err := m.provider.jobRequest(videoAccept, options.Headers)
	if err != nil {
		return nil, err
	}

	if options.Image == nil {
		return apicall.PostJSON(ctx, request, job)
	}

	input, err := m.inputFile(ctx, options.Image)
	if err != nil {
		return nil, err
	}

	encoded, err := json.Marshal(job)
	if err != nil {
		return nil, fmt.Errorf("encode job: %w", err)
	}

	fields := []apicall.FormField{
		{
			Name:      "job",
			Filename:  "job.json",
			MediaType: "application/json",
			Data:      encoded,
		},
		{
			Name:      "input",
			Filename:  "input" + inputExtensions[input.mediaType],
			MediaType: input.mediaType,
			Data:      input.data,
		},
	}

	return apicall.PostForm(ctx, request, fields)
}

// inputFile returns the image bytes, downloading URL images through the
// guarded downloader.
func (m *VideoModel) inputFile(ctx context.Context, file model.VideoFile) (inputFile, error) {
	switch typed := file.(type) {
	case model.VideoBytes:
		return inputFile{
			data:      typed.Data,
			mediaType: typed.MediaType,
		}, nil

	case model.VideoURL:
		downloaded, err := download.Fetch(ctx, download.Request{
			URL:    typed.URL,
			Client: m.provider.httpClient,
		})
		if err != nil {
			return inputFile{}, err
		}

		return inputFile{
			data:      downloaded.Data,
			mediaType: downloadedMediaType(downloaded.MediaType),
		}, nil

	default:
		return inputFile{}, fmt.Errorf("%w: video input %T", model.ErrUnsupported, file)
	}
}

// downloadedMediaType returns the media type of a Content-Type header
// without its parameters.
func downloadedMediaType(contentType string) string {
	mediaType, _, err := mime.ParseMediaType(contentType)
	if err != nil || mediaType == "" {
		return unknownMediaType
	}

	return mediaType
}

// videoJobResult reads the job record and the video from the multipart
// response. The video is the output part, or else a part with a video
// content type.
func videoJobResult(response *apicall.Response) (job, model.VideoDataBytes, error) {
	parts, err := readJobParts(response.Headers, response.Body)
	if err != nil {
		return job{}, model.VideoDataBytes{}, err
	}

	var record *job
	var video *model.VideoDataBytes

	for _, part := range parts {
		isVideo := strings.HasPrefix(part.contentType, "video/")

		switch {
		case part.name == "job":
			decoded, err := decodeJob(part)
			if err != nil {
				return job{}, model.VideoDataBytes{}, err
			}

			record = &decoded

		case part.name == "output":
			mediaType := defaultVideoMediaType
			if isVideo {
				mediaType = part.contentType
			}

			video = &model.VideoDataBytes{
				Data:      part.body,
				MediaType: mediaType,
			}

		case isVideo:
			video = &model.VideoDataBytes{
				Data:      part.body,
				MediaType: part.contentType,
			}
		}
	}

	if record == nil {
		return job{}, model.VideoDataBytes{}, fmt.Errorf("%w: Prodia multipart response missing job part", model.ErrInvalidResponse)
	}

	if video == nil {
		return job{}, model.VideoDataBytes{}, fmt.Errorf("%w: Prodia multipart response missing output video", model.ErrInvalidResponse)
	}

	return *record, *video, nil
}
