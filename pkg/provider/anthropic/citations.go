package anthropic

import (
	"strconv"

	"github.com/aarock1234/ai/pkg/model"
)

const untitledDocument = "Untitled Document"

// citationDocument describes a document that citations may refer to by
// index: a prompt file with citations enabled, or a fetched web page.
type citationDocument struct {
	title     string
	filename  string
	mediaType string
}

// citationDocuments returns the user file parts that have citations
// enabled, in prompt order, which is the order citation indices use.
func citationDocuments(prompt []model.Message) ([]citationDocument, error) {
	var documents []citationDocument

	for _, message := range prompt {
		user, ok := message.(model.UserMessage)
		if !ok {
			continue
		}

		for _, part := range user.Content {
			file, ok := part.(model.FilePart)
			if !ok || !isCitableMediaType(file.MediaType) {
				continue
			}

			options, err := filePartOptions(file.ProviderOptions)
			if err != nil {
				return nil, err
			}

			if options.Citations == nil || !options.Citations.Enabled {
				continue
			}

			title := file.Filename
			if title == "" {
				title = untitledDocument
			}

			documents = append(documents, citationDocument{
				title:     title,
				filename:  file.Filename,
				mediaType: file.MediaType,
			})
		}
	}

	return documents, nil
}

func isCitableMediaType(mediaType string) bool {
	return mediaType == "application/pdf" || mediaType == "text/plain"
}

// citationsMetadata returns the text metadata that carries the native
// citations, or nil when there are none.
func citationsMetadata(citations []Citation) model.ProviderMetadata {
	if len(citations) == 0 {
		return nil
	}

	return model.ProviderMetadata{
		Namespace: TextMetadata{
			Citations: citations,
		},
	}
}

// mapCitations converts native citations into citations of the text they
// annotate.
func mapCitations(citations []Citation, documents []citationDocument) []model.Citation {
	if len(citations) == 0 {
		return nil
	}

	mapped := make([]model.Citation, 0, len(citations))
	for _, citation := range citations {
		mapped = append(mapped, mapCitation(citation, documents))
	}

	return mapped
}

func mapCitation(citation Citation, documents []citationDocument) model.Citation {
	metadata := model.ProviderMetadata{
		Namespace: citation,
	}

	if isURLCitation(citation.Type) {
		url := citation.URL
		if citation.Type == CitationSearchResultLocation {
			url = citation.Source
		}

		return model.Citation{
			Source: model.Source{
				SourceType:       model.SourceURL,
				ID:               url,
				URL:              url,
				Title:            valueOrZero(citation.Title),
				ProviderMetadata: metadata,
			},
			CitedText: citation.CitedText,
		}
	}

	documentIndex := valueOrZero(citation.DocumentIndex)
	document, _ := documentAt(documents, citation.DocumentIndex)

	id := strconv.Itoa(documentIndex)
	if citation.FileID != nil {
		id = *citation.FileID
	}

	mediaType := document.mediaType
	if mediaType == "" {
		mediaType = "text/plain"
	}

	return model.Citation{
		Source: model.Source{
			SourceType:       model.SourceDocument,
			ID:               id,
			MediaType:        mediaType,
			Title:            citedDocumentTitle(citation, document, documentIndex),
			Filename:         document.filename,
			ProviderMetadata: metadata,
		},
		CitedText: citation.CitedText,
	}
}

func isURLCitation(citationType CitationType) bool {
	return citationType == CitationWebSearchResultLocation || citationType == CitationSearchResultLocation
}

func citedDocumentTitle(citation Citation, document citationDocument, documentIndex int) string {
	if citation.DocumentTitle != nil {
		return *citation.DocumentTitle
	}

	if document.title != "" {
		return document.title
	}

	return "Document " + strconv.Itoa(documentIndex)
}

// citationSource returns the source part announced for a citation: a URL
// source for web search citations and a document source for page and
// character locations of known documents.
func citationSource(citation Citation, documents []citationDocument, generateID func() string) (model.Source, bool) {
	if citation.Type == CitationWebSearchResultLocation {
		return model.Source{
			SourceType: model.SourceURL,
			ID:         generateID(),
			URL:        citation.URL,
			Title:      valueOrZero(citation.Title),
			ProviderMetadata: model.ProviderMetadata{
				Namespace: CitationSourceMetadata{
					CitedText:      citation.CitedText,
					EncryptedIndex: citation.EncryptedIndex,
				},
			},
		}, true
	}

	if citation.Type != CitationPageLocation && citation.Type != CitationCharLocation {
		return model.Source{}, false
	}

	document, found := documentAt(documents, citation.DocumentIndex)
	if !found {
		return model.Source{}, false
	}

	title := document.title
	if citation.DocumentTitle != nil {
		title = *citation.DocumentTitle
	}

	return model.Source{
		SourceType: model.SourceDocument,
		ID:         generateID(),
		MediaType:  document.mediaType,
		Title:      title,
		Filename:   document.filename,
		ProviderMetadata: model.ProviderMetadata{
			Namespace: documentLocationMetadata(citation),
		},
	}, true
}

func documentLocationMetadata(citation Citation) CitationSourceMetadata {
	if citation.Type == CitationPageLocation {
		return CitationSourceMetadata{
			CitedText:       citation.CitedText,
			StartPageNumber: citation.StartPageNumber,
			EndPageNumber:   citation.EndPageNumber,
		}
	}

	return CitationSourceMetadata{
		CitedText:      citation.CitedText,
		StartCharIndex: citation.StartCharIndex,
		EndCharIndex:   citation.EndCharIndex,
	}
}

func documentAt(documents []citationDocument, index *int) (citationDocument, bool) {
	if index == nil || *index < 0 || *index >= len(documents) {
		return citationDocument{}, false
	}

	return documents[*index], true
}
