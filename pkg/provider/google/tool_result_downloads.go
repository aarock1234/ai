package google

import (
	"context"
	"regexp"
	"slices"

	"github.com/aarock1234/ai/internal/download"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/pkg/model"
)

// toolResultDownloads downloads tool result file URLs that Vertex AI cannot
// reference, so they are sent inline.
type toolResultDownloads struct {
	maxBytes int64

	// supportedURLs are the URLs the model reads itself.
	supportedURLs map[string][]*regexp.Regexp
}

// apply returns the prompt with downloaded tool result files. The caller's
// messages are not modified.
func (d toolResultDownloads) apply(ctx context.Context, prompt []model.Message) ([]model.Message, error) {
	converted := make([]model.Message, 0, len(prompt))

	for _, message := range prompt {
		switch message := message.(type) {
		case model.AssistantMessage:
			content, err := d.assistantContent(ctx, message.Content)
			if err != nil {
				return nil, err
			}

			message.Content = content
			converted = append(converted, message)

		case model.ToolMessage:
			content, err := d.toolContent(ctx, message.Content)
			if err != nil {
				return nil, err
			}

			message.Content = content
			converted = append(converted, message)

		default:
			converted = append(converted, message)
		}
	}

	return converted, nil
}

func (d toolResultDownloads) assistantContent(ctx context.Context, parts []model.AssistantPart) ([]model.AssistantPart, error) {
	converted := slices.Clone(parts)

	for index, assistantPart := range converted {
		result, ok := assistantPart.(model.ToolResultPart)
		if !ok {
			continue
		}

		output, err := d.output(ctx, result.Output)
		if err != nil {
			return nil, err
		}

		result.Output = output
		converted[index] = result
	}

	return converted, nil
}

func (d toolResultDownloads) toolContent(ctx context.Context, parts []model.ToolPart) ([]model.ToolPart, error) {
	converted := slices.Clone(parts)

	for index, toolPart := range converted {
		result, ok := toolPart.(model.ToolResultPart)
		if !ok {
			continue
		}

		output, err := d.output(ctx, result.Output)
		if err != nil {
			return nil, err
		}

		result.Output = output
		converted[index] = result
	}

	return converted, nil
}

// output downloads the unsupported file URLs of a content output.
func (d toolResultDownloads) output(ctx context.Context, output model.ToolResultOutput) (model.ToolResultOutput, error) {
	content, ok := output.(model.ContentOutput)
	if !ok {
		return output, nil
	}

	items := slices.Clone(content.Value)

	for index, item := range items {
		file, ok := item.(model.ToolResultFile)
		if !ok {
			continue
		}

		downloaded, err := d.file(ctx, file)
		if err != nil {
			return nil, err
		}

		items[index] = downloaded
	}

	return model.ContentOutput{
		Value: items,
	}, nil
}

// file downloads a file URL unless the model reads it itself. The media type
// is detected from image signatures, then taken from the response when the
// declared type is not a full type.
func (d toolResultDownloads) file(ctx context.Context, file model.ToolResultFile) (model.ToolResultFile, error) {
	data, ok := file.Data.(model.FileURL)
	if !ok {
		return file, nil
	}

	location := data.URL.String()
	if isURLSupported(location, file.MediaType, d.supportedURLs) {
		return file, nil
	}

	downloaded, err := download.Fetch(ctx, download.Request{
		URL:      location,
		MaxBytes: d.maxBytes,
	})
	if err != nil {
		return model.ToolResultFile{}, err
	}

	mediaType := file.MediaType
	if detected, ok := mediatype.Detect(downloaded.Data, "image"); ok {
		mediaType = detected
	} else if downloaded.MediaType != "" && !mediatype.IsFull(file.MediaType) {
		mediaType = downloaded.MediaType
	}

	file.MediaType = mediaType
	file.Data = model.FileBytes{
		Data: downloaded.Data,
	}

	return file, nil
}
