package gateway

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/url"

	"github.com/aarock1234/ai/pkg/model"
)

// File data types of the specification's file data union.
const (
	fileDataBytes     = "data"
	fileDataURL       = "url"
	fileDataReference = "reference"
	fileDataText      = "text"
)

// wireMetadata is provider metadata on the wire: provider namespaces
// mapped to provider-specific objects.
type wireMetadata map[string]json.RawMessage

// providerMetadata converts wire metadata. The values stay raw JSON, which
// model.ProviderValue decodes into the provider's types.
func (m wireMetadata) providerMetadata() model.ProviderMetadata {
	if len(m) == 0 {
		return nil
	}

	metadata := make(model.ProviderMetadata, len(m))
	for namespace, value := range m {
		metadata[namespace] = value
	}

	return metadata
}

// wireWarning is a call warning on the wire.
type wireWarning struct {
	Type    model.WarningType `json:"type"`
	Feature string            `json:"feature"`
	Details string            `json:"details"`
	Setting string            `json:"setting"`
	Message string            `json:"message"`
}

// decodeWarnings converts wire warnings.
func decodeWarnings(warnings []wireWarning) []model.Warning {
	if len(warnings) == 0 {
		return nil
	}

	converted := make([]model.Warning, 0, len(warnings))
	for _, warning := range warnings {
		converted = append(converted, model.Warning{
			Type:    warning.Type,
			Feature: warning.Feature,
			Details: warning.Details,
			Setting: warning.Setting,
			Message: warning.Message,
		})
	}

	return converted
}

// wireFileData is the file data union on the wire. Inline bytes travel as
// base64.
type wireFileData struct {
	Type      string            `json:"type"`
	Data      *string           `json:"data,omitempty"`
	URL       *string           `json:"url,omitempty"`
	Reference map[string]string `json:"reference,omitempty"`
	Text      *string           `json:"text,omitempty"`
}

// encodeFileData converts file data for the wire.
func encodeFileData(data model.FileData) (wireFileData, error) {
	switch data := data.(type) {
	case model.FileBytes:
		return wireFileData{
			Type: fileDataBytes,
			Data: new(base64.StdEncoding.EncodeToString(data.Data)),
		}, nil

	case model.FileURL:
		if data.URL == nil {
			return wireFileData{}, fmt.Errorf("%w: file URL is nil", model.ErrInvalidPrompt)
		}

		return wireFileData{
			Type: fileDataURL,
			URL:  new(data.URL.String()),
		}, nil

	case model.FileReference:
		return wireFileData{
			Type:      fileDataReference,
			Reference: data.IDs,
		}, nil

	case model.FileText:
		return wireFileData{
			Type: fileDataText,
			Text: new(data.Text),
		}, nil

	default:
		return wireFileData{}, fmt.Errorf("%w: file data %T", model.ErrInvalidPrompt, data)
	}
}

// decodeFileData converts generated file data, which is inline base64 or a
// URL.
func decodeFileData(data wireFileData) (model.FileData, error) {
	switch {
	case data.Type == fileDataBytes && data.Data != nil:
		decoded, err := base64.StdEncoding.DecodeString(*data.Data)
		if err != nil {
			return nil, fmt.Errorf("%w: decode base64 file data: %w", model.ErrInvalidResponse, err)
		}

		return model.FileBytes{
			Data: decoded,
		}, nil

	case data.Type == fileDataURL && data.URL != nil:
		parsed, err := url.Parse(*data.URL)
		if err != nil {
			return nil, fmt.Errorf("%w: file URL: %w", model.ErrInvalidResponse, err)
		}

		return model.FileURL{
			URL: parsed,
		}, nil

	default:
		return nil, fmt.Errorf("%w: unsupported generated file data type %q", model.ErrInvalidResponse, data.Type)
	}
}

// wireUsage is language model usage on the wire.
type wireUsage struct {
	InputTokens  wireInputTokens  `json:"inputTokens"`
	OutputTokens wireOutputTokens `json:"outputTokens"`
	Raw          json.RawMessage  `json:"raw,omitempty"`
}

type wireInputTokens struct {
	Total      *int `json:"total"`
	NoCache    *int `json:"noCache"`
	CacheRead  *int `json:"cacheRead"`
	CacheWrite *int `json:"cacheWrite"`
}

type wireOutputTokens struct {
	Total     *int `json:"total"`
	Text      *int `json:"text"`
	Reasoning *int `json:"reasoning"`
}

func (u wireUsage) usage() model.Usage {
	return model.Usage{
		InputTokens: model.InputTokens{
			Total:      u.InputTokens.Total,
			NoCache:    u.InputTokens.NoCache,
			CacheRead:  u.InputTokens.CacheRead,
			CacheWrite: u.InputTokens.CacheWrite,
		},
		OutputTokens: model.OutputTokens{
			Total:     u.OutputTokens.Total,
			Text:      u.OutputTokens.Text,
			Reasoning: u.OutputTokens.Reasoning,
		},
		Raw: u.Raw,
	}
}

// wireFinishReason is a finish reason on the wire.
type wireFinishReason struct {
	Unified model.UnifiedFinishReason `json:"unified"`
	Raw     *string                   `json:"raw"`
}

func (r wireFinishReason) finishReason() model.FinishReason {
	return model.FinishReason{
		Unified: r.Unified,
		Raw:     stringValue(r.Raw),
	}
}

// wireSource is a source on the wire, as content or in a citation.
type wireSource struct {
	SourceType       model.SourceType `json:"sourceType"`
	ID               string           `json:"id"`
	URL              string           `json:"url"`
	Title            string           `json:"title"`
	MediaType        string           `json:"mediaType"`
	Filename         string           `json:"filename"`
	ProviderMetadata wireMetadata     `json:"providerMetadata"`
}

func (s wireSource) source() model.Source {
	return model.Source{
		SourceType:       s.SourceType,
		ID:               s.ID,
		URL:              s.URL,
		Title:            s.Title,
		MediaType:        s.MediaType,
		Filename:         s.Filename,
		ProviderMetadata: s.ProviderMetadata.providerMetadata(),
	}
}

// wireCitation links text to a source on the wire.
type wireCitation struct {
	Source     wireSource `json:"source"`
	StartIndex *int       `json:"startIndex"`
	EndIndex   *int       `json:"endIndex"`
	CitedText  string     `json:"citedText"`
}

func decodeCitations(citations []wireCitation) []model.Citation {
	if len(citations) == 0 {
		return nil
	}

	converted := make([]model.Citation, 0, len(citations))
	for _, citation := range citations {
		converted = append(converted, model.Citation{
			Source:     citation.Source.source(),
			StartIndex: citation.StartIndex,
			EndIndex:   citation.EndIndex,
			CitedText:  citation.CitedText,
		})
	}

	return converted
}

// typeHeader reads the discriminator of a tagged union member.
type typeHeader struct {
	Type string `json:"type"`
}

// decodeAs decodes raw into a T, reporting malformed members as invalid
// responses.
func decodeAs[T any](raw json.RawMessage) (T, error) {
	var value T
	if err := json.Unmarshal(raw, &value); err != nil {
		return value, fmt.Errorf("%w: %w", model.ErrInvalidResponse, err)
	}

	return value, nil
}
