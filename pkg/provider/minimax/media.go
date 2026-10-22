package minimax

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/anthropic"
)

// Frame sampling rate limits of video parts, in frames per second.
const (
	minVideoFPS = 0.2
	maxVideoFPS = 5.0
)

// fileReferencePrefix marks MiniMax file IDs in video URLs.
const fileReferencePrefix = "mm_file://"

// ImageDetail is the image processing detail of M3.1 and later models.
type ImageDetail string

// Image detail levels.
const (
	ImageDetailLow     ImageDetail = "low"
	ImageDetailDefault ImageDetail = "default"
	ImageDetailHigh    ImageDetail = "high"
)

// ImagePartOptions are the provider options of image file parts sent to
// M3.1 and later models, read from the "minimax" namespace.
type ImagePartOptions struct {
	// Detail is the image processing detail. Empty uses the server default.
	Detail ImageDetail `json:"detail,omitempty"`
}

// VideoPartOptions are the provider options of video file parts sent to
// M3.1 and later models, read from the "minimax" namespace.
type VideoPartOptions struct {
	// FPS is the frame sampling rate, from 0.2 to 5 frames per second.
	FPS *float64 `json:"fps,omitempty"`
}

// mediaSource is the source of an image or video block, with the part's
// processing options.
type mediaSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`

	Detail ImageDetail `json:"detail,omitempty"`
	FPS    *float64    `json:"fps,omitempty"`
}

// convertMediaPart sends image and video parts of M3.1 and later models as
// image and video blocks. Other files use the Anthropic conversion.
func convertMediaPart(part model.FilePart) (*anthropic.MediaContent, error) {
	var contentType anthropic.MediaContentType

	switch mediatype.TopLevel(part.MediaType) {
	case "image":
		contentType = anthropic.MediaContentImage
	case "video":
		contentType = anthropic.MediaContentVideo
	default:
		return nil, nil
	}

	source, err := mediaSourceOf(part, contentType)
	if err != nil {
		return nil, err
	}

	if err := applyMediaOptions(&source, part.ProviderOptions, contentType); err != nil {
		return nil, err
	}

	encoded, err := json.Marshal(source)
	if err != nil {
		return nil, fmt.Errorf("encode minimax %s source: %w", contentType, err)
	}

	return &anthropic.MediaContent{
		Type:   contentType,
		Source: encoded,
	}, nil
}

// mediaSourceOf sends inline data as base64 and URLs as they are. Video
// file references become MiniMax file URLs.
func mediaSourceOf(part model.FilePart, contentType anthropic.MediaContentType) (mediaSource, error) {
	switch data := part.Data.(type) {
	case model.FileBytes:
		mediaType, err := mediatype.ResolveFile(part.MediaType, data)
		if err != nil {
			return mediaSource{}, err
		}

		return mediaSource{
			Type:      "base64",
			MediaType: mediaType,
			Data:      base64.StdEncoding.EncodeToString(data.Data),
		}, nil

	case model.FileURL:
		if data.URL == nil {
			return mediaSource{}, fmt.Errorf("%w: file URL is missing", model.ErrInvalidPrompt)
		}

		return mediaSource{
			Type: "url",
			URL:  data.URL.String(),
		}, nil

	case model.FileReference:
		return fileReferenceSource(data, contentType)

	default:
		return mediaSource{}, fmt.Errorf("%w: MiniMax %s input with %s data", model.ErrUnsupported, contentType, fileDataType(part.Data))
	}
}

// fileReferenceSource returns the MiniMax file URL of a video reference.
// Image references are not supported.
func fileReferenceSource(reference model.FileReference, contentType anthropic.MediaContentType) (mediaSource, error) {
	if contentType != anthropic.MediaContentVideo {
		return mediaSource{}, fmt.Errorf("%w: MiniMax image file references", model.ErrUnsupported)
	}

	fileID, found := reference.IDs[Namespace]
	if !found {
		return mediaSource{}, fmt.Errorf("%w: file reference has no %s file ID", model.ErrInvalidPrompt, Namespace)
	}

	if !strings.HasPrefix(fileID, fileReferencePrefix) {
		fileID = fileReferencePrefix + fileID
	}

	return mediaSource{
		Type: "url",
		URL:  fileID,
	}, nil
}

// applyMediaOptions validates and adds the image detail or video frame rate
// of a part.
func applyMediaOptions(source *mediaSource, options model.ProviderOptions, contentType anthropic.MediaContentType) error {
	if contentType == anthropic.MediaContentImage {
		imageOptions, err := readImagePartOptions(options)
		if err != nil {
			return err
		}

		source.Detail = imageOptions.Detail

		return nil
	}

	videoOptions, err := readVideoPartOptions(options)
	if err != nil {
		return err
	}

	source.FPS = videoOptions.FPS

	return nil
}

func readImagePartOptions(options model.ProviderOptions) (ImagePartOptions, error) {
	imageOptions, _, err := model.ProviderValue[ImagePartOptions](options, Namespace)
	if err != nil {
		return ImagePartOptions{}, fmt.Errorf("minimax image part options: %w", err)
	}

	switch imageOptions.Detail {
	case "", ImageDetailLow, ImageDetailDefault, ImageDetailHigh:
		return imageOptions, nil
	default:
		return ImagePartOptions{}, fmt.Errorf(
			"%w: minimax image part options: unsupported detail %q",
			model.ErrInvalidArgument,
			imageOptions.Detail,
		)
	}
}

func readVideoPartOptions(options model.ProviderOptions) (VideoPartOptions, error) {
	videoOptions, _, err := model.ProviderValue[VideoPartOptions](options, Namespace)
	if err != nil {
		return VideoPartOptions{}, fmt.Errorf("minimax video part options: %w", err)
	}

	fps := videoOptions.FPS
	if fps != nil && (*fps < minVideoFPS || *fps > maxVideoFPS) {
		return VideoPartOptions{}, fmt.Errorf(
			"%w: minimax video part options: fps %v must be between %v and %v",
			model.ErrInvalidArgument,
			*fps,
			minVideoFPS,
			maxVideoFPS,
		)
	}

	return videoOptions, nil
}

// fileDataType names the kind of file data in errors.
func fileDataType(data model.FileData) string {
	switch data.(type) {
	case model.FileText:
		return "text"
	default:
		return fmt.Sprintf("%T", data)
	}
}
