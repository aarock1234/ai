package wirejson

import (
	"encoding/base64"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// FileData is encoded file content: a member of the provider
// specification's tagged file data union, or a flattened value. A nil
// FileData encodes as null.
type FileData interface {
	isFileData()
}

// TaggedFileBytes is inline file content, base64 encoded.
type TaggedFileBytes struct {
	Type string `json:"type"`
	Data string `json:"data"`
}

// TaggedFileURL is file content behind a URL.
type TaggedFileURL struct {
	Type string `json:"type"`
	URL  string `json:"url"`
}

// TaggedFileReference is file content stored with providers, keyed by
// provider name.
type TaggedFileReference struct {
	Type      string            `json:"type"`
	Reference map[string]string `json:"reference"`
}

// TaggedFileText is file content given as text.
type TaggedFileText struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// FlatFileString is flattened file content: base64 bytes, a URL or text.
type FlatFileString string

// FlatFileReference is a flattened provider reference map.
type FlatFileReference map[string]string

func (TaggedFileBytes) isFileData()     {}
func (TaggedFileURL) isFileData()       {}
func (TaggedFileReference) isFileData() {}
func (TaggedFileText) isFileData()      {}
func (FlatFileString) isFileData()      {}
func (FlatFileReference) isFileData()   {}

// fileData encodes file content in the encoder's file data encoding.
// Telemetry, policies and recordings report content that cannot be encoded,
// such as a URL file without a URL, as null rather than failing.
func (e Encoder) fileData(data model.FileData) FileData {
	encode := FlattenFileData
	if e.FileData == TaggedFileData {
		encode = TagFileData
	}

	encoded, err := encode(data)
	if err != nil {
		return nil
	}

	return encoded
}

// FlattenFileData encodes file content as a base64 string, a URL, text or a
// provider reference map.
func FlattenFileData(data model.FileData) (FileData, error) {
	switch typed := data.(type) {
	case model.FileBytes:
		return FlatFileString(base64.StdEncoding.EncodeToString(typed.Data)), nil

	case model.FileURL:
		if typed.URL == nil {
			return nil, missingFileURL()
		}

		return FlatFileString(typed.URL.String()), nil

	case model.FileText:
		return FlatFileString(typed.Text), nil

	case model.FileReference:
		return FlatFileReference(typed.IDs), nil

	default:
		return nil, unsupportedFileData(data)
	}
}

// TagFileData encodes file content as a member of the tagged union, such
// as {"type":"data","data":"<base64>"}.
func TagFileData(data model.FileData) (FileData, error) {
	switch typed := data.(type) {
	case model.FileBytes:
		return TaggedFileBytes{
			Type: "data",
			Data: base64.StdEncoding.EncodeToString(typed.Data),
		}, nil

	case model.FileURL:
		if typed.URL == nil {
			return nil, missingFileURL()
		}

		return TaggedFileURL{
			Type: "url",
			URL:  typed.URL.String(),
		}, nil

	case model.FileText:
		return TaggedFileText{
			Type: "text",
			Text: typed.Text,
		}, nil

	case model.FileReference:
		return TaggedFileReference{
			Type:      "reference",
			Reference: typed.IDs,
		}, nil

	default:
		return nil, unsupportedFileData(data)
	}
}

func missingFileURL() error {
	return fmt.Errorf("%w: file data URL is missing", model.ErrInvalidPrompt)
}

func unsupportedFileData(data model.FileData) error {
	return fmt.Errorf("%w: file data %T", model.ErrUnsupported, data)
}
