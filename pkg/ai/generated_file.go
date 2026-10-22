package ai

import (
	"encoding/base64"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

// GeneratedFile is a file produced by a model, such as an image.
type GeneratedFile struct {
	Data []byte

	// MediaType is the IANA media type of the file.
	MediaType string

	// ProviderMetadata is provider-specific data about this file.
	ProviderMetadata model.ProviderMetadata
}

// Base64 returns the file content encoded as standard base64.
func (f GeneratedFile) Base64() string {
	return base64.StdEncoding.EncodeToString(f.Data)
}

// GeneratedAudioFile is generated speech audio.
type GeneratedAudioFile struct {
	GeneratedFile

	// Format is the audio format, such as "mp3" or "wav".
	Format string
}

// newGeneratedAudioFile derives the audio format from the media type:
// "audio/mpeg" and media types without a subtype are "mp3"; any other subtype
// is used as is.
func newGeneratedAudioFile(data []byte, mediaType string) GeneratedAudioFile {
	format := "mp3"

	_, subtype, found := strings.Cut(mediaType, "/")
	if found && subtype != "" && mediaType != "audio/mpeg" {
		format = subtype
	}

	return GeneratedAudioFile{
		Data:      data,
		MediaType: mediaType,
		Format:    format,
	}
}
