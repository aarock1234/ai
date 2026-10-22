package ai

import (
	"context"
	"fmt"
	"regexp"
	"slices"
	"strings"

	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/internal/merge"
	"github.com/aarock1234/ai/pkg/model"
)

// convertPromptOptions describe the conversion of a standardized prompt into
// the prompt of one model call.
type convertPromptOptions struct {
	prompt standardizedPrompt

	// supportedURLs is the model's SupportedURLs. File URLs that do not match
	// are downloaded and sent inline.
	supportedURLs map[string][]*regexp.Regexp

	// download resolves file URLs. Nil uses the default downloader.
	download DownloadFunction
}

// downloadedAssets maps a file URL to its downloaded content.
type downloadedAssets map[string]*DownloadedFile

// convertToModelPrompt converts a standardized prompt into model messages: it
// prepends the instructions, downloads file URLs the model does not support,
// decodes data URLs, resolves file media types, removes approval requests and
// non-provider approval responses, merges consecutive tool messages, and
// verifies that every application tool call has a result.
func convertToModelPrompt(ctx context.Context, options convertPromptOptions) ([]model.Message, error) {
	download := options.download
	if download == nil {
		download = defaultDownloadFunction(NewDownloader(DownloadOptions{}))
	}

	assets, err := downloadAssets(ctx, options.prompt.messages, download, options.supportedURLs)
	if err != nil {
		return nil, err
	}

	messages := make([]model.Message, 0, len(options.prompt.instructions)+len(options.prompt.messages))
	for _, instruction := range options.prompt.instructions {
		messages = append(messages, instruction)
	}

	for _, message := range options.prompt.messages {
		converted, err := convertMessage(message, assets)
		if err != nil {
			return nil, err
		}

		messages = append(messages, converted)
	}

	combined := combineToolMessages(messages)

	if err := checkToolResults(combined); err != nil {
		return nil, err
	}

	return removeEmptyToolMessages(combined), nil
}

func convertMessage(message model.Message, assets downloadedAssets) (model.Message, error) {
	switch typed := message.(type) {
	case model.SystemMessage:
		return typed, nil

	case model.UserMessage:
		return convertUserMessage(typed, assets)

	case model.AssistantMessage:
		return convertAssistantMessage(typed, assets)

	case model.ToolMessage:
		return convertToolMessage(typed, assets)

	case nil:
		return nil, &InvalidPromptError{
			Message: "message is nil",
		}

	default:
		return nil, &InvalidMessageRoleError{
			Role: message.Role(),
		}
	}
}

// convertUserMessage removes empty text parts and resolves file parts.
func convertUserMessage(message model.UserMessage, assets downloadedAssets) (model.UserMessage, error) {
	content := make([]model.UserPart, 0, len(message.Content))

	for _, part := range message.Content {
		switch typed := part.(type) {
		case model.TextPart:
			if typed.Text != "" {
				content = append(content, typed)
			}

		case model.FilePart:
			resolved, err := resolveFilePart(typed, assets)
			if err != nil {
				return model.UserMessage{}, err
			}

			content = append(content, resolved)

		default:
			return model.UserMessage{}, &InvalidPromptError{
				Message: fmt.Sprintf("unsupported user message part %T", part),
			}
		}
	}

	return model.UserMessage{
		Content:         content,
		ProviderOptions: message.ProviderOptions,
	}, nil
}

// convertAssistantMessage removes empty text parts and tool approval
// requests, decodes data URLs and converts tool result outputs.
func convertAssistantMessage(message model.AssistantMessage, assets downloadedAssets) (model.AssistantMessage, error) {
	content := make([]model.AssistantPart, 0, len(message.Content))

	for _, part := range message.Content {
		converted, keep, err := convertAssistantPart(part, assets)
		if err != nil {
			return model.AssistantMessage{}, err
		}

		if keep {
			content = append(content, converted)
		}
	}

	return model.AssistantMessage{
		Content:         content,
		ProviderOptions: message.ProviderOptions,
	}, nil
}

func convertAssistantPart(part model.AssistantPart, assets downloadedAssets) (model.AssistantPart, bool, error) {
	switch typed := part.(type) {
	case model.TextPart:
		isEmpty := typed.Text == "" && typed.ProviderOptions == nil

		return typed, !isEmpty, nil

	case model.ToolApprovalRequestPart:
		return nil, false, nil

	case model.FilePart:
		data, mediaType, err := normalizeFileData(typed.Data, typed.MediaType)
		if err != nil {
			return nil, false, err
		}

		typed.Data = data
		typed.MediaType = mediaType

		return typed, true, nil

	case model.ReasoningFilePart:
		data, mediaType, err := normalizeFileData(typed.Data, typed.MediaType)
		if err != nil {
			return nil, false, err
		}

		if !isInlineOrURL(data) {
			return nil, false, &InvalidPromptError{
				Message: fmt.Sprintf("unsupported reasoning file data type %T", data),
			}
		}

		typed.Data = data
		typed.MediaType = mediaType

		return typed, true, nil

	case model.ToolResultPart:
		output, err := convertToolResultOutput(typed.Output, assets)
		if err != nil {
			return nil, false, err
		}

		typed.Output = output

		return typed, true, nil

	case model.ReasoningPart, model.CustomPart, model.ToolCallPart:
		return typed, true, nil

	default:
		return nil, false, &InvalidPromptError{
			Message: fmt.Sprintf("unsupported assistant message part %T", part),
		}
	}
}

// convertToolMessage converts tool result outputs and keeps approval
// responses only for provider-executed tools, which the provider resumes.
func convertToolMessage(message model.ToolMessage, assets downloadedAssets) (model.ToolMessage, error) {
	content := make([]model.ToolPart, 0, len(message.Content))

	for _, part := range message.Content {
		switch typed := part.(type) {
		case model.ToolResultPart:
			output, err := convertToolResultOutput(typed.Output, assets)
			if err != nil {
				return model.ToolMessage{}, err
			}

			typed.Output = output
			content = append(content, typed)

		case model.ToolApprovalResponsePart:
			if typed.ProviderExecuted {
				content = append(content, typed)
			}

		default:
			return model.ToolMessage{}, &InvalidPromptError{
				Message: fmt.Sprintf("unsupported tool message part %T", part),
			}
		}
	}

	return model.ToolMessage{
		Content:         content,
		ProviderOptions: message.ProviderOptions,
	}, nil
}

// convertToolResultOutput resolves the files of a multipart tool output.
func convertToolResultOutput(output model.ToolResultOutput, assets downloadedAssets) (model.ToolResultOutput, error) {
	multipart, ok := output.(model.ContentOutput)
	if !ok {
		return output, nil
	}

	items := make([]model.ToolResultContent, 0, len(multipart.Value))

	for _, item := range multipart.Value {
		file, isFile := item.(model.ToolResultFile)
		if !isFile {
			items = append(items, item)

			continue
		}

		data, mediaType, err := resolveFileContent(file.Data, file.MediaType, assets)
		if err != nil {
			return nil, err
		}

		file.Data = data
		file.MediaType = mediaType
		items = append(items, file)
	}

	return model.ContentOutput{
		Value: items,
	}, nil
}

func resolveFilePart(part model.FilePart, assets downloadedAssets) (model.FilePart, error) {
	data, mediaType, err := resolveFileContent(part.Data, part.MediaType, assets)
	if err != nil {
		return model.FilePart{}, err
	}

	part.Data = data
	part.MediaType = mediaType

	return part, nil
}

// resolveFileContent decodes data URLs, replaces downloaded URLs with their
// content and determines a media type: a downloaded Content-Type completes a
// wildcard media type, and inline content refines it from its signature.
func resolveFileContent(data model.FileData, mediaType string, assets downloadedAssets) (model.FileData, string, error) {
	data, mediaType, err := normalizeFileData(data, mediaType)
	if err != nil {
		return nil, "", err
	}

	if fileURL, ok := data.(model.FileURL); ok {
		if downloaded := assets[fileURL.URL.String()]; downloaded != nil {
			data = model.FileBytes{
				Data: downloaded.Data,
			}

			if downloaded.MediaType != "" && !mediatype.IsFull(mediaType) {
				mediaType = downloaded.MediaType
			}
		}
	}

	if inline, ok := data.(model.FileBytes); ok {
		mediaType = detectInlineMediaType(inline.Data, mediaType)
	}

	if mediaType == "" {
		return nil, "", &InvalidPromptError{
			Message: "media type is missing for file part",
		}
	}

	return data, mediaType, nil
}

// detectInlineMediaType prefers a detected image signature, as upstream does,
// and otherwise completes a wildcard media type from the content.
func detectInlineMediaType(data []byte, mediaType string) string {
	if detected, ok := mediatype.Detect(data, "image"); ok {
		return detected
	}

	if mediaType == "" || mediatype.IsFull(mediaType) {
		return mediaType
	}

	if resolved, ok := mediatype.Resolve(mediaType, data); ok {
		return resolved
	}

	return mediaType
}

// normalizeFileData decodes data URLs into inline content, taking the media
// type from the data URL when it declares one.
func normalizeFileData(data model.FileData, mediaType string) (model.FileData, string, error) {
	fileURL, isURL := data.(model.FileURL)

	switch {
	case data == nil:
		return nil, "", &InvalidDataContentError{
			Message: "file data is missing",
		}

	case isURL && fileURL.URL == nil:
		return nil, "", &InvalidDataContentError{
			Message: "file URL is missing",
		}

	case !isURL || fileURL.URL.Scheme != "data":
		return data, mediaType, nil
	}

	dataURLMediaType, content, err := dataurl.Parse(fileURL.URL.String())
	if err != nil {
		return nil, "", &InvalidDataContentError{
			Content: fileURL.URL.String(),
			Message: "invalid data URL",
			Cause:   err,
		}
	}

	if dataURLMediaType != "" {
		mediaType = dataURLMediaType
	}

	inline := model.FileBytes{
		Data: content,
	}

	return inline, mediaType, nil
}

func isInlineOrURL(data model.FileData) bool {
	switch data.(type) {
	case model.FileBytes, model.FileURL:
		return true
	default:
		return false
	}
}

// downloadAssets downloads the file URLs of user messages and tool results,
// keyed by URL. The download function decides which URLs to fetch; by default
// it fetches those the model does not support.
func downloadAssets(
	ctx context.Context,
	messages []model.Message,
	download DownloadFunction,
	supportedURLs map[string][]*regexp.Regexp,
) (downloadedAssets, error) {
	requests, err := plannedDownloads(messages, supportedURLs)
	if err != nil {
		return nil, err
	}

	if len(requests) == 0 {
		return nil, nil
	}

	files, err := download(ctx, requests)
	if err != nil {
		return nil, err
	}

	if len(files) != len(requests) {
		return nil, fmt.Errorf("download function returned %d results for %d requests", len(files), len(requests))
	}

	assets := make(downloadedAssets, len(files))
	for index, file := range files {
		if file != nil {
			assets[requests[index].URL.String()] = file
		}
	}

	return assets, nil
}

// plannedDownloads lists the URL files of the prompt in message order.
func plannedDownloads(messages []model.Message, supportedURLs map[string][]*regexp.Regexp) ([]DownloadRequest, error) {
	var requests []DownloadRequest

	for _, file := range promptFiles(messages) {
		data, _, err := normalizeFileData(file.data, file.mediaType)
		if err != nil {
			return nil, err
		}

		fileURL, ok := data.(model.FileURL)
		if !ok {
			continue
		}

		isSupported := file.mediaType != "" && isURLSupported(urlSupportCheck{
			mediaType:     file.mediaType,
			url:           fileURL.URL.String(),
			supportedURLs: supportedURLs,
		})

		requests = append(requests, DownloadRequest{
			URL:                   fileURL.URL,
			IsURLSupportedByModel: isSupported,
		})
	}

	return requests, nil
}

// promptFile is a file that may need downloading.
type promptFile struct {
	data      model.FileData
	mediaType string
}

// promptFiles returns the files of user messages and the files inside tool
// results of assistant and tool messages.
func promptFiles(messages []model.Message) []promptFile {
	var files []promptFile

	for _, message := range messages {
		switch typed := message.(type) {
		case model.UserMessage:
			for _, part := range typed.Content {
				if file, ok := part.(model.FilePart); ok {
					files = append(files, promptFile{
						data:      file.Data,
						mediaType: file.MediaType,
					})
				}
			}

		case model.AssistantMessage:
			for _, part := range typed.Content {
				if result, ok := part.(model.ToolResultPart); ok {
					files = append(files, toolResultFiles(result.Output)...)
				}
			}

		case model.ToolMessage:
			for _, part := range typed.Content {
				if result, ok := part.(model.ToolResultPart); ok {
					files = append(files, toolResultFiles(result.Output)...)
				}
			}
		}
	}

	return files
}

func toolResultFiles(output model.ToolResultOutput) []promptFile {
	multipart, ok := output.(model.ContentOutput)
	if !ok {
		return nil
	}

	var files []promptFile
	for _, item := range multipart.Value {
		if file, ok := item.(model.ToolResultFile); ok {
			files = append(files, promptFile{
				data:      file.Data,
				mediaType: file.MediaType,
			})
		}
	}

	return files
}

// urlSupportCheck asks whether a model downloads a URL itself.
type urlSupportCheck struct {
	// mediaType may be a full type, a wildcard such as "image/*" or a bare
	// top-level type such as "image".
	mediaType string

	url           string
	supportedURLs map[string][]*regexp.Regexp
}

// isURLSupported reports whether the URL matches a pattern registered for
// its media type or for every media type. Media types compare
// case-insensitively. A bare top-level media type matches only its wildcard
// entry, because the exact subtype is unknown.
func isURLSupported(check urlSupportCheck) bool {
	mediaType := strings.ToLower(check.mediaType)
	isTopLevelOnly := !strings.Contains(mediaType, "/")

	for key, patterns := range check.supportedURLs {
		prefix := supportedMediaTypePrefix(key)

		if !mediaTypeMatchesPrefix(mediaType, prefix, isTopLevelOnly) {
			continue
		}

		matches := slices.ContainsFunc(patterns, func(pattern *regexp.Regexp) bool {
			return pattern != nil && pattern.MatchString(check.url)
		})
		if matches {
			return true
		}
	}

	return false
}

// supportedMediaTypePrefix turns a SupportedURLs key into a lowercase media
// type prefix: "*" and "*/*" match everything, and "image/*" becomes
// "image/".
func supportedMediaTypePrefix(key string) string {
	key = strings.ToLower(key)
	if key == "*" || key == "*/*" {
		return ""
	}

	return strings.Replace(key, "*", "", 1)
}

func mediaTypeMatchesPrefix(mediaType, prefix string, isTopLevelOnly bool) bool {
	switch {
	case prefix == "":
		return true

	case isTopLevelOnly:
		return mediaType+"/" == prefix

	case strings.HasSuffix(prefix, "/"):
		return strings.HasPrefix(mediaType, prefix)

	default:
		return mediaType == prefix
	}
}

// combineToolMessages merges consecutive tool messages into one. When the
// earlier message has provider options, they move onto its last part so that
// they keep applying to the same content.
func combineToolMessages(messages []model.Message) []model.Message {
	combined := make([]model.Message, 0, len(messages))

	for _, message := range messages {
		toolMessage, isTool := message.(model.ToolMessage)
		if !isTool || len(combined) == 0 {
			combined = append(combined, message)

			continue
		}

		previous, previousIsTool := combined[len(combined)-1].(model.ToolMessage)
		if !previousIsTool {
			combined = append(combined, message)

			continue
		}

		combined[len(combined)-1] = appendToolMessage(previous, toolMessage)
	}

	return combined
}

func appendToolMessage(previous, next model.ToolMessage) model.ToolMessage {
	content := slices.Clone(previous.Content)
	hadContent := len(content) > 0

	if hadContent && previous.ProviderOptions != nil {
		last := len(content) - 1
		content[last] = withToolPartOptions(content[last], previous.ProviderOptions)
	}

	options := next.ProviderOptions
	if !hadContent {
		options = merge.Deep(previous.ProviderOptions, next.ProviderOptions)
	}

	return model.ToolMessage{
		Content:         append(content, next.Content...),
		ProviderOptions: options,
	}
}

// withToolPartOptions applies message-level provider options beneath the
// part's own options.
func withToolPartOptions(part model.ToolPart, messageOptions model.ProviderOptions) model.ToolPart {
	switch typed := part.(type) {
	case model.ToolResultPart:
		typed.ProviderOptions = merge.Deep(messageOptions, typed.ProviderOptions)

		return typed

	case model.ToolApprovalResponsePart:
		typed.ProviderOptions = merge.Deep(messageOptions, typed.ProviderOptions)

		return typed

	default:
		return part
	}
}

// checkToolResults verifies that every tool call the application must run
// has a result before the next user or system message and at the end of the
// prompt.
func checkToolResults(messages []model.Message) error {
	var pending []string

	for _, message := range messages {
		switch typed := message.(type) {
		case model.AssistantMessage:
			for _, part := range typed.Content {
				call, isCall := part.(model.ToolCallPart)
				if !isCall || call.ProviderExecuted {
					continue
				}

				if !slices.Contains(pending, call.ToolCallID) {
					pending = append(pending, call.ToolCallID)
				}
			}

		case model.ToolMessage:
			for _, part := range typed.Content {
				result, isResult := part.(model.ToolResultPart)
				if !isResult {
					continue
				}

				pending = slices.DeleteFunc(pending, func(pendingID string) bool {
					return pendingID == result.ToolCallID
				})
			}

		case model.UserMessage, model.SystemMessage:
			if len(pending) > 0 {
				return &MissingToolResultsError{
					ToolCallIDs: pending,
				}
			}
		}
	}

	if len(pending) > 0 {
		return &MissingToolResultsError{
			ToolCallIDs: pending,
		}
	}

	return nil
}

// removeEmptyToolMessages drops tool messages left empty after removing
// approval responses, which providers would reject.
func removeEmptyToolMessages(messages []model.Message) []model.Message {
	return slices.DeleteFunc(messages, func(message model.Message) bool {
		toolMessage, isTool := message.(model.ToolMessage)

		return isTool && len(toolMessage.Content) == 0
	})
}
