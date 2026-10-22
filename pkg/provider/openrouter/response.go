package openrouter

import (
	"encoding/base64"
	"fmt"
	"net/url"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

// responseExtensions holds the OpenRouter members of a response or chunk that
// the OpenAI-compatible model does not read.
type responseExtensions struct {
	Choices []extensionChoice `json:"choices"`
}

type extensionChoice struct {
	// Message is set in responses and Delta in stream chunks.
	Message *messageExtensions `json:"message"`
	Delta   *messageExtensions `json:"delta"`
}

type messageExtensions struct {
	ReasoningDetails []ReasoningDetail `json:"reasoning_details"`
	Images           []generatedImage  `json:"images"`
	Annotations      []annotation      `json:"annotations"`
}

// first returns the extensions of the first choice's message or delta.
func (r responseExtensions) first() messageExtensions {
	if len(r.Choices) == 0 {
		return messageExtensions{}
	}

	choice := r.Choices[0]
	switch {
	case choice.Message != nil:
		return *choice.Message
	case choice.Delta != nil:
		return *choice.Delta
	default:
		return messageExtensions{}
	}
}

// generatedImage is an image produced by an image generation model.
type generatedImage struct {
	Type     string    `json:"type"`
	ImageURL *imageURL `json:"image_url"`
}

type imageURL struct {
	URL string `json:"url"`
}

// annotation is a message annotation. Web search results are url_citation
// annotations.
type annotation struct {
	Type        string       `json:"type"`
	URLCitation *urlCitation `json:"url_citation"`
}

type urlCitation struct {
	URL        string `json:"url"`
	Title      string `json:"title"`
	Content    string `json:"content"`
	StartIndex *int   `json:"start_index"`
	EndIndex   *int   `json:"end_index"`
}

// CitationMetadata is the provider metadata of a web citation source.
type CitationMetadata struct {
	// Content is the cited passage of the page.
	Content string `json:"content,omitempty"`

	// StartIndex and EndIndex locate the citation in the message text.
	StartIndex *int `json:"startIndex,omitempty"`
	EndIndex   *int `json:"endIndex,omitempty"`
}

// imageFiles converts generated images into files. Data URLs become inline
// bytes and other URLs are kept as URLs.
func imageFiles(images []generatedImage) ([]model.File, error) {
	var files []model.File

	for _, image := range images {
		if image.ImageURL == nil {
			continue
		}

		file, err := imageFile(image.ImageURL.URL)
		if err != nil {
			return nil, err
		}

		files = append(files, file)
	}

	return files, nil
}

func imageFile(address string) (model.File, error) {
	if !strings.HasPrefix(address, "data:") {
		parsed, err := url.Parse(address)
		if err != nil {
			return model.File{}, fmt.Errorf("%w: generated image URL: %w", model.ErrInvalidResponse, err)
		}

		return model.File{
			MediaType: "image/*",
			Data: model.FileURL{
				URL: parsed,
			},
		}, nil
	}

	mediaType, data, err := decodeDataURL(address)
	if err != nil {
		return model.File{}, err
	}

	return model.File{
		MediaType: mediaType,
		Data: model.FileBytes{
			Data: data,
		},
	}, nil
}

// decodeDataURL reads a base64 data URL such as "data:image/png;base64,...".
func decodeDataURL(address string) (string, []byte, error) {
	header, payload, found := strings.Cut(strings.TrimPrefix(address, "data:"), ",")
	mediaType, isBase64 := strings.CutSuffix(header, ";base64")

	if !found || !isBase64 {
		return "", nil, fmt.Errorf("%w: generated image is not a base64 data URL", model.ErrInvalidResponse)
	}

	data, err := base64.StdEncoding.DecodeString(payload)
	if err != nil {
		return "", nil, fmt.Errorf("%w: decode generated image: %w", model.ErrInvalidResponse, err)
	}

	return mediaType, data, nil
}

// citationSources converts url_citation annotations into sources. Other
// annotation types are skipped.
func citationSources(annotations []annotation, generateID func() string) []model.Source {
	var sources []model.Source

	for _, annotation := range annotations {
		citation := annotation.URLCitation
		if annotation.Type != "url_citation" || citation == nil {
			continue
		}

		sources = append(sources, model.Source{
			SourceType: model.SourceURL,
			ID:         generateID(),
			URL:        citation.URL,
			Title:      citation.Title,
			ProviderMetadata: model.ProviderMetadata{
				Namespace: CitationMetadata{
					Content:    citation.Content,
					StartIndex: citation.StartIndex,
					EndIndex:   citation.EndIndex,
				},
			},
		})
	}

	return sources
}
