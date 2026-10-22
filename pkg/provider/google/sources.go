package google

import (
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

const (
	unknownDocumentTitle = "Unknown Document"
	octetStreamMediaType = "application/octet-stream"
)

// documentMediaTypes maps file extensions of retrieved documents to media
// types.
var documentMediaTypes = []struct {
	extension string
	mediaType string
}{
	{
		extension: ".pdf",
		mediaType: "application/pdf",
	},
	{
		extension: ".txt",
		mediaType: "text/plain",
	},
	{
		extension: ".docx",
		mediaType: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
	},
	{
		extension: ".doc",
		mediaType: "application/msword",
	},
	{
		extension: ".md",
		mediaType: "text/markdown",
	},
	{
		extension: ".markdown",
		mediaType: "text/markdown",
	},
}

// extractSources converts grounding chunks into sources: web pages, image
// search results, Google Maps places and retrieved documents.
func extractSources(grounding *GroundingMetadata, generateID func() string) []model.Source {
	if grounding == nil {
		return nil
	}

	var sources []model.Source
	for _, chunk := range grounding.GroundingChunks {
		source, ok := chunkSource(chunk, generateID)
		if ok {
			sources = append(sources, source)
		}
	}

	return sources
}

func chunkSource(chunk GroundingChunk, generateID func() string) (model.Source, bool) {
	switch {
	case chunk.Web != nil:
		return urlSource(chunk.Web.URI, chunk.Web.Title, generateID), true

	case chunk.Image != nil:
		// Google requires attribution to the source page, not the image.
		return urlSource(chunk.Image.SourceURI, chunk.Image.Title, generateID), true

	case chunk.RetrievedContext != nil:
		return retrievedContextSource(*chunk.RetrievedContext, generateID)

	case chunk.Maps != nil:
		if chunk.Maps.URI == nil || *chunk.Maps.URI == "" {
			return model.Source{}, false
		}

		return urlSource(*chunk.Maps.URI, chunk.Maps.Title, generateID), true

	default:
		return model.Source{}, false
	}
}

// retrievedContextSource converts retrieval results. HTTP URIs are URL
// sources; other URIs and file search stores are document sources.
func retrievedContextSource(retrieved RetrievedContextChunk, generateID func() string) (model.Source, bool) {
	uri := valueOrEmpty(retrieved.URI)

	if strings.HasPrefix(uri, "http://") || strings.HasPrefix(uri, "https://") {
		return urlSource(uri, retrieved.Title, generateID), true
	}

	title := valueOrEmpty(retrieved.Title)
	if retrieved.Title == nil {
		title = unknownDocumentTitle
	}

	if uri != "" {
		return documentSource(title, documentMediaType(uri), lastPathSegment(uri), generateID), true
	}

	store := valueOrEmpty(retrieved.FileSearchStore)
	if store != "" {
		return documentSource(title, octetStreamMediaType, lastPathSegment(store), generateID), true
	}

	return model.Source{}, false
}

func urlSource(url string, title *string, generateID func() string) model.Source {
	return model.Source{
		SourceType: model.SourceURL,
		ID:         generateID(),
		URL:        url,
		Title:      valueOrEmpty(title),
	}
}

func documentSource(title string, mediaType string, filename string, generateID func() string) model.Source {
	return model.Source{
		SourceType: model.SourceDocument,
		ID:         generateID(),
		Title:      title,
		MediaType:  mediaType,
		Filename:   filename,
	}
}

func documentMediaType(uri string) string {
	for _, document := range documentMediaTypes {
		if strings.HasSuffix(uri, document.extension) {
			return document.mediaType
		}
	}

	return octetStreamMediaType
}

func lastPathSegment(path string) string {
	return path[strings.LastIndex(path, "/")+1:]
}

func valueOrEmpty(value *string) string {
	if value == nil {
		return ""
	}

	return *value
}
