package anthropic

import (
	"encoding/base64"
	"fmt"

	"github.com/aarock1234/ai/internal/optional"

	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/pkg/model"
)

// filePartOptions reads the Anthropic options of a file part.
func filePartOptions(options model.ProviderOptions) (FilePartOptions, error) {
	parsed, _, err := model.ProviderValue[FilePartOptions](options, Namespace)

	return parsed, err
}

// userFilePart converts a file part of a user message, first through the
// host's media part converter when one is set.
func (c *promptConverter) userFilePart(part model.FilePart, cacheControl *CacheControl) (requestBlock, error) {
	if c.convertMediaPart == nil {
		return c.filePart(part, cacheControl)
	}

	media, err := c.convertMediaPart(part)
	if err != nil {
		return nil, err
	}

	if media == nil {
		return c.filePart(part, cacheControl)
	}

	return &mediaBlock{
		Type:         media.Type,
		Source:       media.Source,
		CacheControl: cacheControl,
	}, nil
}

func (c *promptConverter) filePart(part model.FilePart, cacheControl *CacheControl) (requestBlock, error) {
	options, err := filePartOptions(part.ProviderOptions)
	if err != nil {
		return nil, err
	}

	switch data := part.Data.(type) {
	case model.FileReference:
		return c.fileReference(part, data, options, cacheControl)

	case model.FileText:
		return c.document(part, options, cacheControl, contentSource{
			Type:      "text",
			MediaType: "text/plain",
			Data:      data.Text,
		}), nil

	case model.FileURL:
		return c.urlFile(part, data, options, cacheControl)

	case model.FileBytes:
		return c.inlineFile(part, data, options, cacheControl)

	default:
		return nil, fmt.Errorf("%w: unsupported file data %T", model.ErrInvalidPrompt, part.Data)
	}
}

func (c *promptConverter) fileReference(
	part model.FilePart,
	data model.FileReference,
	options FilePartOptions,
	cacheControl *CacheControl,
) (requestBlock, error) {
	fileID, err := anthropicFileID(data)
	if err != nil {
		return nil, err
	}

	c.result.betas.add("files-api-2025-04-14")

	if optional.Or(options.ContainerUpload, false) {
		return &containerUploadBlock{
			Type:   blockContainerUpload,
			FileID: fileID,
		}, nil
	}

	source := contentSource{
		Type:   "file",
		FileID: fileID,
	}

	if mediatype.TopLevel(part.MediaType) == "image" {
		return &imageBlock{
			Type:         blockImage,
			Source:       source,
			CacheControl: cacheControl,
		}, nil
	}

	return &documentBlock{
		Type:         blockDocument,
		Source:       source,
		CacheControl: cacheControl,
	}, nil
}

func anthropicFileID(data model.FileReference) (string, error) {
	fileID, found := data.IDs[Namespace]
	if !found {
		return "", fmt.Errorf("%w: file reference has no anthropic file ID", model.ErrInvalidPrompt)
	}

	return fileID, nil
}

func (c *promptConverter) urlFile(
	part model.FilePart,
	data model.FileURL,
	options FilePartOptions,
	cacheControl *CacheControl,
) (requestBlock, error) {
	if data.URL == nil {
		return nil, fmt.Errorf("%w: file URL is missing", model.ErrInvalidPrompt)
	}

	source := contentSource{
		Type: "url",
		URL:  data.URL.String(),
	}

	switch {
	case mediatype.TopLevel(part.MediaType) == "image":
		return &imageBlock{
			Type:         blockImage,
			Source:       source,
			CacheControl: cacheControl,
		}, nil

	case part.MediaType == "application/pdf":
		c.result.betas.add("pdfs-2024-09-25")

		return c.document(part, options, cacheControl, source), nil

	case part.MediaType == "text/plain":
		return c.document(part, options, cacheControl, source), nil

	default:
		return nil, fmt.Errorf("%w: media type: %s", model.ErrUnsupported, part.MediaType)
	}
}

func (c *promptConverter) inlineFile(
	part model.FilePart,
	data model.FileBytes,
	options FilePartOptions,
	cacheControl *CacheControl,
) (requestBlock, error) {
	topLevel := mediatype.TopLevel(part.MediaType)

	if topLevel == "image" {
		mediaType, err := mediatype.ResolveFile(part.MediaType, data)
		if err != nil {
			return nil, err
		}

		return &imageBlock{
			Type: blockImage,
			Source: contentSource{
				Type:      "base64",
				MediaType: mediaType,
				Data:      base64.StdEncoding.EncodeToString(data.Data),
			},
			CacheControl: cacheControl,
		}, nil
	}

	if topLevel == "application" {
		mediaType, err := mediatype.ResolveFile(part.MediaType, data)
		if err != nil {
			return nil, err
		}

		if mediaType == "application/pdf" {
			c.result.betas.add("pdfs-2024-09-25")

			return c.document(part, options, cacheControl, contentSource{
				Type:      "base64",
				MediaType: "application/pdf",
				Data:      base64.StdEncoding.EncodeToString(data.Data),
			}), nil
		}
	}

	if part.MediaType == "text/plain" {
		return c.document(part, options, cacheControl, contentSource{
			Type:      "text",
			MediaType: "text/plain",
			Data:      string(data.Data),
		}), nil
	}

	return nil, fmt.Errorf("%w: media type: %s", model.ErrUnsupported, part.MediaType)
}

// document builds a document block with the part's title, context and
// citation settings.
func (c *promptConverter) document(
	part model.FilePart,
	options FilePartOptions,
	cacheControl *CacheControl,
	source contentSource,
) requestBlock {
	block := &documentBlock{
		Type:         blockDocument,
		Source:       source,
		Title:        optional.Or(options.Title, part.Filename),
		Context:      optional.Or(options.Context, ""),
		CacheControl: cacheControl,
	}

	if options.Citations != nil && options.Citations.Enabled {
		block.Citations = &CitationsConfig{
			Enabled: true,
		}
	}

	return block
}
