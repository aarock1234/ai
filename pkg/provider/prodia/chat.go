package prodia

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"regexp"
	"slices"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/download"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/internal/reasoning"
	"github.com/aarock1234/ai/pkg/model"
)

// languageAccept asks for the job record and the outputs as multipart
// parts.
const languageAccept = "multipart/form-data"

// AspectRatio is the aspect ratio of an output image.
type AspectRatio string

// Aspect ratios.
const (
	AspectRatio1x1  AspectRatio = "1:1"
	AspectRatio2x3  AspectRatio = "2:3"
	AspectRatio3x2  AspectRatio = "3:2"
	AspectRatio4x5  AspectRatio = "4:5"
	AspectRatio5x4  AspectRatio = "5:4"
	AspectRatio4x7  AspectRatio = "4:7"
	AspectRatio7x4  AspectRatio = "7:4"
	AspectRatio9x16 AspectRatio = "9:16"
	AspectRatio16x9 AspectRatio = "16:9"
	AspectRatio9x21 AspectRatio = "9:21"
	AspectRatio21x9 AspectRatio = "21:9"
)

// aspectRatios are the accepted aspect ratios.
var aspectRatios = []AspectRatio{
	AspectRatio1x1,
	AspectRatio2x3,
	AspectRatio3x2,
	AspectRatio4x5,
	AspectRatio5x4,
	AspectRatio4x7,
	AspectRatio7x4,
	AspectRatio9x16,
	AspectRatio16x9,
	AspectRatio9x21,
	AspectRatio21x9,
}

// imageExtensions are the file extensions of input images.
var imageExtensions = map[string]string{
	"image/png":  ".png",
	"image/jpeg": ".jpg",
	"image/webp": ".webp",
}

// ChatOptions are the provider options of language models, read from the
// "prodia" namespace.
type ChatOptions struct {
	// AspectRatio is the aspect ratio of the output image.
	AspectRatio *AspectRatio `json:"aspectRatio,omitempty"`
}

// ChatModel is a Prodia multimodal language model. It sends the text
// and the first image of the last user message, prefixed by the system
// message, as one job.
type ChatModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "prodia.language" by default.
func (m *ChatModel) Provider() string {
	return m.provider.name + ".language"
}

// ModelID returns the model identifier.
func (m *ChatModel) ModelID() string {
	return m.modelID
}

// SupportedURLs returns no patterns: image URLs are downloaded and sent
// inline.
func (m *ChatModel) SupportedURLs() map[string][]*regexp.Regexp {
	return map[string][]*regexp.Regexp{}
}

type languageJobRequest struct {
	Type   string            `json:"type"`
	Config languageJobConfig `json:"config"`
}

type languageJobConfig struct {
	Prompt          string       `json:"prompt"`
	IncludeMessages bool         `json:"include_messages"`
	AspectRatio     *AspectRatio `json:"aspect_ratio,omitempty"`
}

// inputImage is the image sent with the job.
type inputImage struct {
	mediaType string
	data      []byte
}

// Generate runs one job and returns its text and images.
func (m *ChatModel) Generate(ctx context.Context, options model.CallOptions) (*model.GenerateResult, error) {
	timestamp := time.Now()
	warnings := unsupportedLanguageSettings(options)

	chatOptions, err := readChatOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	image, err := m.lastUserImage(ctx, options.Prompt)
	if err != nil {
		return nil, err
	}

	fields, err := m.jobForm(jobPrompt(options.Prompt), chatOptions, image)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.jobRequest(languageAccept, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostForm(ctx, request, fields)
	if err != nil {
		return nil, err
	}

	record, content, err := languageJobResult(response)
	if err != nil {
		return nil, err
	}

	return &model.GenerateResult{
		Content: content,
		FinishReason: model.FinishReason{
			Unified: model.FinishStop,
		},
		ProviderMetadata: model.ProviderMetadata{
			Namespace: jobMetadata(record),
		},
		Warnings: warnings,
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: timestamp,
				ModelID:   m.modelID,
			},
			Headers: response.Headers,
		},
	}, nil
}

// Stream runs one job and replays its result as a stream.
func (m *ChatModel) Stream(ctx context.Context, options model.CallOptions) (*model.StreamResult, error) {
	result, err := m.Generate(ctx, options)
	if err != nil {
		return nil, err
	}

	stream := model.NewStreamResult(m.replay(result), nil)
	stream.Response = model.StreamResponseInfo{
		Headers: result.Response.Headers,
	}

	return stream, nil
}

// replay yields the parts of a completed result.
func (m *ChatModel) replay(result *model.GenerateResult) iter.Seq[model.StreamPart] {
	return func(yield func(model.StreamPart) bool) {
		parts := []model.StreamPart{
			model.StreamStart{
				Warnings: result.Warnings,
			},
			result.Response.ResponseMetadata,
		}

		for _, content := range result.Content {
			parts = append(parts, m.contentParts(content)...)
		}

		parts = append(parts, model.Finish{
			Usage:            result.Usage,
			FinishReason:     result.FinishReason,
			ProviderMetadata: result.ProviderMetadata,
		})

		for _, part := range parts {
			if !yield(part) {
				return
			}
		}
	}
}

func (m *ChatModel) contentParts(content model.Content) []model.StreamPart {
	switch content := content.(type) {
	case model.Text:
		textID := m.provider.generateID()

		return []model.StreamPart{
			model.TextStart{
				ID: textID,
			},
			model.TextDelta{
				ID:    textID,
				Delta: content.Text,
			},
			model.TextEnd{
				ID: textID,
			},
		}

	case model.File:
		return []model.StreamPart{
			content,
		}

	default:
		return nil
	}
}

func unsupportedLanguageSettings(options model.CallOptions) []model.Warning {
	settings := []struct {
		feature string
		set     bool
	}{
		{
			feature: "temperature",
			set:     options.Temperature != nil,
		},
		{
			feature: "topP",
			set:     options.TopP != nil,
		},
		{
			feature: "topK",
			set:     options.TopK != nil,
		},
		{
			feature: "seed",
			set:     options.Seed != nil,
		},
		{
			feature: "maxOutputTokens",
			set:     options.MaxOutputTokens != nil,
		},
		{
			feature: "stopSequences",
			set:     options.StopSequences != nil,
		},
		{
			feature: "presencePenalty",
			set:     options.PresencePenalty != nil,
		},
		{
			feature: "frequencyPenalty",
			set:     options.FrequencyPenalty != nil,
		},
		{
			feature: "tools",
			set:     len(options.Tools) > 0,
		},
		{
			feature: "toolChoice",
			set:     options.ToolChoice != nil,
		},
		{
			feature: "responseFormat",
			set:     options.ResponseFormat != nil && options.ResponseFormat.Type != model.ResponseFormatText,
		},
	}

	var warnings []model.Warning
	for _, setting := range settings {
		if setting.set {
			warnings = append(warnings, model.UnsupportedWarning(setting.feature, ""))
		}
	}

	if reasoning.IsCustom(options.Reasoning) {
		warnings = append(warnings, model.UnsupportedWarning(
			"reasoning",
			"This provider does not support reasoning configuration.",
		))
	}

	return warnings
}

func readChatOptions(options model.ProviderOptions) (ChatOptions, error) {
	chatOptions, _, err := model.ProviderValue[ChatOptions](options, Namespace)
	if err != nil {
		return ChatOptions{}, err
	}

	if chatOptions.AspectRatio != nil && !slices.Contains(aspectRatios, *chatOptions.AspectRatio) {
		return ChatOptions{}, fmt.Errorf("%w: prodia aspectRatio %q", model.ErrInvalidArgument, *chatOptions.AspectRatio)
	}

	return chatOptions, nil
}

// jobPrompt joins the text of the last user message, prefixed by the last
// system message.
func jobPrompt(prompt []model.Message) string {
	var system string
	for _, message := range prompt {
		if message, ok := message.(model.SystemMessage); ok {
			system = message.Content
		}
	}

	var texts []string
	if user, ok := lastUserMessage(prompt); ok {
		for _, part := range user.Content {
			if text, ok := part.(model.TextPart); ok {
				texts = append(texts, text.Text)
			}
		}
	}

	text := strings.Join(texts, "\n")
	if system != "" {
		return system + "\n" + text
	}

	return text
}

func lastUserMessage(prompt []model.Message) (model.UserMessage, bool) {
	for _, message := range slices.Backward(prompt) {
		if user, ok := message.(model.UserMessage); ok {
			return user, true
		}
	}

	return model.UserMessage{}, false
}

// lastUserImage returns the first image of the last user message, if any.
func (m *ChatModel) lastUserImage(ctx context.Context, prompt []model.Message) (*inputImage, error) {
	user, ok := lastUserMessage(prompt)
	if !ok {
		return nil, nil
	}

	for _, part := range user.Content {
		file, ok := part.(model.FilePart)
		if !ok || mediatype.TopLevel(file.MediaType) != "image" {
			continue
		}

		data, err := m.imageData(ctx, file.Data)
		if err != nil {
			return nil, err
		}

		return &inputImage{
			mediaType: imageMediaType(file.MediaType, data),
			data:      data,
		}, nil
	}

	return nil, nil
}

// imageData returns inline image bytes, downloading URL images.
func (m *ChatModel) imageData(ctx context.Context, data model.FileData) ([]byte, error) {
	switch data := data.(type) {
	case model.FileBytes:
		return data.Data, nil

	case model.FileURL:
		downloaded, err := download.Fetch(ctx, download.Request{
			URL:    data.URL.String(),
			Client: m.provider.httpClient,
		})
		if err != nil {
			return nil, err
		}

		return downloaded.Data, nil

	case model.FileReference:
		return nil, fmt.Errorf("%w: file parts with provider references", model.ErrUnsupported)

	case model.FileText:
		return nil, fmt.Errorf("%w: text file parts", model.ErrUnsupported)

	default:
		return nil, fmt.Errorf("%w: file data %T", model.ErrUnsupported, data)
	}
}

// imageMediaType returns the full media type of an input image. Undetected
// types default to PNG.
func imageMediaType(declared string, data []byte) string {
	if mediatype.IsFull(declared) {
		return declared
	}

	if detected, ok := mediatype.Detect(data, mediatype.TopLevel(declared)); ok {
		return detected
	}

	return "image/png"
}

// jobForm builds the multipart job: the job record and the optional input
// image.
func (m *ChatModel) jobForm(prompt string, chatOptions ChatOptions, image *inputImage) ([]apicall.FormField, error) {
	encoded, err := json.Marshal(languageJobRequest{
		Type: m.modelID,
		Config: languageJobConfig{
			Prompt:          prompt,
			IncludeMessages: true,
			AspectRatio:     chatOptions.AspectRatio,
		},
	})
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
	}

	if image != nil {
		fields = append(fields, apicall.FormField{
			Name:      "input",
			Filename:  "input" + imageExtensions[image.mediaType],
			MediaType: image.mediaType,
			Data:      image.data,
		})
	}

	return fields, nil
}

// languageJobResult reads the job record, text and images from the
// multipart response.
func languageJobResult(response *apicall.Response) (job, []model.Content, error) {
	parts, err := readJobParts(response.Headers, response.Body)
	if err != nil {
		return job{}, nil, err
	}

	var record *job
	var text *string
	var files []model.Content

	for _, part := range parts {
		switch {
		case part.name == "job":
			decoded, err := decodeJob(part)
			if err != nil {
				return job{}, nil, err
			}

			record = &decoded

		case part.name == "output" && isTextOutput(part):
			text = new(string(part.body))

		case part.name == "output" && isImageOutput(part):
			files = append(files, model.File{
				MediaType: part.contentType,
				Data: model.FileBytes{
					Data: part.body,
				},
			})
		}
	}

	if record == nil {
		return job{}, nil, fmt.Errorf("%w: Prodia multipart response missing job part", model.ErrInvalidResponse)
	}

	var content []model.Content
	if text != nil {
		content = append(content, model.Text{
			Text: *text,
		})
	}

	return *record, append(content, files...), nil
}

var _ model.LanguageModel = (*ChatModel)(nil)
