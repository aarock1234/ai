package openai

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

// EncodeRequest converts a normalized model.Request into the OpenAI
// chat completions wire format.
func EncodeRequest(modelID string, req *model.Request, stream bool) (*ChatRequest, error) {
	msgs, err := convertMessages(req.Messages)
	if err != nil {
		return nil, err
	}

	cr := &ChatRequest{
		Model:            modelID,
		Messages:         msgs,
		Temperature:      req.Temperature,
		MaxTokens:        req.MaxTokens,
		TopP:             req.TopP,
		PresencePenalty:  req.PresencePenalty,
		FrequencyPenalty: req.FrequencyPenalty,
		Seed:             req.Seed,
		Stop:             req.StopSequences,
		Stream:           stream,
	}

	metadata, err := convertProviderMetadata(req.ProviderOptions)
	if err != nil {
		return nil, err
	}
	if len(metadata) > 0 {
		cr.Metadata = metadata
	}

	if stream {
		cr.StreamOptions = &streamOptions{IncludeUsage: true}
	}

	if len(req.Tools) > 0 {
		cr.Tools = convertTools(req.Tools)
	}

	if req.ToolChoice != nil {
		cr.ToolChoice = convertToolChoice(req.ToolChoice)
	}

	if req.ResponseFormat != nil {
		cr.ResponseFormat = convertResponseFormat(req.ResponseFormat)
	}

	if len(req.Modalities) > 0 {
		cr.Modalities = make([]string, len(req.Modalities))
		for i, m := range req.Modalities {
			cr.Modalities[i] = string(m)
		}
	}

	if req.Audio != nil {
		cr.Audio = &audioConfig{
			Voice:  req.Audio.Voice,
			Format: req.Audio.Format,
		}
	}

	if req.Reasoning != nil {
		cr.Reasoning = &reasoningConfig{
			Effort:    req.Reasoning.Effort,
			MaxTokens: req.Reasoning.MaxTokens,
			Exclude:   req.Reasoning.Exclude,
		}
	}

	if req.Prediction != nil && req.Prediction.Content != "" {
		cr.Prediction = &predictionConfig{
			Type:    "content",
			Content: req.Prediction.Content,
		}
	}

	if len(req.Headers) > 0 {
		cr.Headers = req.Headers.Clone()
	}

	return cr, nil
}

// DecodeResponse converts an OpenAI chat completions response into
// the normalized model.Response.
func DecodeResponse(resp *ChatResponse) *model.Response {
	if len(resp.Choices) == 0 {
		return &model.Response{
			FinishReason: model.FinishReasonUnknown,
			Usage:        decodeUsage(&resp.Usage),
			Metadata:     responseMetadata(resp),
		}
	}

	choice := resp.Choices[0]
	content := decodeMessageContent(&choice.Message)

	result := &model.Response{
		Content:         content,
		FinishReason:    mapFinishReason(choice.FinishReason),
		RawFinishReason: choice.FinishReason,
		Usage:           decodeUsage(&resp.Usage),
		Metadata:        responseMetadata(resp),
	}

	if choice.Message.Refusal != "" {
		result.Warnings = append(result.Warnings, "refusal: "+choice.Message.Refusal)
	}

	return result
}

// decodeMessageContent assembles the ordered content slice for an
// assistant message. The order favors reasoning first (it runs before
// the user-facing answer), then text, then attachments and citations,
// then tool calls last since they terminate the response.
func decodeMessageContent(msg *chatMessage) []model.Content {
	var content []model.Content

	// Prefer the structured reasoning_details array when present; fall
	// back to the flat reasoning/reasoning_content strings otherwise.
	switch {
	case len(msg.ReasoningDetails) > 0:
		for _, part := range msg.ReasoningDetails {
			if block, ok := reasoningContentFromPart(part); ok {
				content = append(content, block)
			}
		}
	case msg.Reasoning != "":
		content = append(content, model.ReasoningContent{Text: msg.Reasoning})
	case msg.ReasoningContent != "":
		content = append(content, model.ReasoningContent{Text: msg.ReasoningContent})
	}

	if text := msg.Content.TextValue(); text != "" {
		content = append(content, model.TextContent{Text: text})
	}

	for _, image := range msg.Images {
		if image.ImageURL == nil {
			continue
		}
		if file, ok := fileContentFromDataURL(image.ImageURL.URL); ok {
			content = append(content, file)
		}
	}

	if audio, ok := fileContentFromAudio(msg.Audio); ok {
		content = append(content, audio)
		if msg.Audio.Transcript != "" {
			content = append(content, model.TextContent{Text: msg.Audio.Transcript})
		}
	}

	for _, annotation := range msg.Annotations {
		if source, ok := sourceContentFromAnnotation(annotation); ok {
			content = append(content, source)
		}
	}

	for _, tc := range msg.ToolCalls {
		content = append(content, model.ToolCallContent{
			ToolCallID: tc.ID,
			ToolName:   tc.Function.Name,
			Args:       tc.Function.Arguments,
		})
	}

	return content
}

// fileContentFromAudio decodes an audio message payload into a
// FileContent block. It returns false when the payload is missing or
// malformed so callers can skip it.
func fileContentFromAudio(audio *wireMessageAudio) (model.FileContent, bool) {
	if audio == nil || audio.Data == "" {
		return model.FileContent{}, false
	}

	data, err := base64.StdEncoding.DecodeString(audio.Data)
	if err != nil {
		return model.FileContent{}, false
	}

	file := model.FileContent{
		Data:     data,
		MIMEType: mimeTypeForAudioFormat(audio.Format),
	}
	if audio.ID != "" {
		file.Metadata.Provider = map[string]string{
			"audio_id": audio.ID,
		}
	}

	return file, true
}

func reasoningContentFromPart(part wireReasoningPart) (model.ReasoningContent, bool) {
	block := model.ReasoningContent{
		Signature: part.Signature,
		ID:        part.ID,
		Format:    part.Format,
	}

	switch part.Type {
	case "reasoning.encrypted":
		block.Redacted = true
	case "reasoning.summary":
		block.Text = part.Summary
	default:
		// reasoning.text and unknown variants both carry plain text.
		block.Text = part.Text
	}

	// Drop parts that carry neither text nor a redaction marker.
	if block.Text == "" && !block.Redacted && block.Signature == "" {
		return model.ReasoningContent{}, false
	}

	return block, true
}

func fileContentFromDataURL(url string) (model.FileContent, bool) {
	const prefix = "data:"
	if !strings.HasPrefix(url, prefix) {
		return model.FileContent{
			URL: url,
		}, url != ""
	}

	header, payload, ok := strings.Cut(url[len(prefix):], ",")
	if !ok {
		return model.FileContent{}, false
	}

	mime, encoding, ok := strings.Cut(header, ";")
	if !ok {
		return model.FileContent{}, false
	}

	if !strings.EqualFold(strings.TrimSpace(encoding), "base64") {
		return model.FileContent{}, false
	}

	data, err := base64.StdEncoding.DecodeString(payload)
	if err != nil {
		return model.FileContent{}, false
	}

	return model.FileContent{
		Data:     data,
		MIMEType: mime,
	}, true
}

func sourceContentFromAnnotation(a wireAnnotation) (model.SourceContent, bool) {
	switch {
	case a.URLCitation != nil && a.URLCitation.URL != "":
		return model.SourceContent{
			SourceType: model.SourceTypeURL,
			URL:        a.URLCitation.URL,
			Title:      a.URLCitation.Title,
			StartIndex: a.URLCitation.StartIndex,
			EndIndex:   a.URLCitation.EndIndex,
		}, true
	case a.FileCitation != nil && a.FileCitation.FileID != "":
		return model.SourceContent{
			SourceType: model.SourceTypeDocument,
			ID:         a.FileCitation.FileID,
			Filename:   a.FileCitation.Filename,
			Title:      a.FileCitation.Title,
		}, true
	default:
		return model.SourceContent{}, false
	}
}

func mimeTypeForAudioFormat(format string) string {
	if format == "" {
		return "audio/wav"
	}

	switch strings.ToLower(format) {
	case "wav":
		return "audio/wav"
	case "mp3":
		return "audio/mpeg"
	case "flac":
		return "audio/flac"
	case "opus":
		return "audio/opus"
	case "pcm16":
		return "audio/pcm"
	case "aac":
		return "audio/aac"
	default:
		return "audio/" + strings.ToLower(format)
	}
}

func decodeUsage(u *wireUsage) model.Usage {
	if u == nil {
		return model.Usage{}
	}

	usage := model.Usage{
		InputTokens:  u.PromptTokens,
		OutputTokens: u.CompletionTokens,
	}

	if u.PromptTokensDetails != nil {
		usage.CachedInputTokens = u.PromptTokensDetails.CachedTokens
		usage.AudioInputTokens = u.PromptTokensDetails.AudioTokens
		usage.ImageInputTokens = u.PromptTokensDetails.ImageTokens
	}

	if u.CompletionTokensDetails != nil {
		usage.ReasoningTokens = u.CompletionTokensDetails.ReasoningTokens
		usage.AudioOutputTokens = u.CompletionTokensDetails.AudioTokens
		usage.ImageOutputTokens = u.CompletionTokensDetails.ImageTokens
	}

	return usage
}

func convertProviderMetadata(opts model.ProviderOptions) (map[string]string, error) {
	raw, ok := opts.Lookup("openai", "metadata")
	if !ok {
		return nil, nil
	}

	var metadata map[string]string
	if err := json.Unmarshal(raw, &metadata); err != nil {
		return nil, fmt.Errorf("openai metadata option must be an object of string values: %w", err)
	}

	return metadata, nil
}

func responseMetadata(resp *ChatResponse) model.ProviderMetadata {
	metadata := model.ProviderMetadata{
		RequestID: resp.ID,
		ModelID:   resp.Model,
	}

	if len(resp.Metadata) > 0 {
		metadata.Provider = make(map[string]string, len(resp.Metadata))
		for key, value := range resp.Metadata {
			metadata.Provider[key] = value
		}
	}

	return metadata
}

// convertMessages translates model messages into the OpenAI wire format.
func convertMessages(msgs []model.Message) ([]chatMessage, error) {
	result := make([]chatMessage, 0, len(msgs))

	for _, msg := range msgs {
		converted, err := convertMessage(msg)
		if err != nil {
			return nil, err
		}

		result = append(result, converted...)
	}

	return result, nil
}

// convertMessage converts a single model.Message. A message with
// tool call parts becomes an assistant message; tool result parts
// each become separate tool-role messages.
func convertMessage(msg model.Message) ([]chatMessage, error) {
	var (
		text             string
		content          []contentPart
		toolCalls        []wireToolCall
		toolResults      []chatMessage
		reasoningDetails []wireReasoningPart
		reasoningText    strings.Builder
	)

	for _, part := range msg.Parts {
		switch p := part.(type) {
		case model.TextPart:
			text += p.Text
			content = append(content, contentPart{
				Type: "text",
				Text: p.Text,
			})
		case model.ReasoningPart:
			detail, ok := reasoningDetailFromPart(p)
			if ok {
				reasoningDetails = append(reasoningDetails, detail)
			}
			if !p.Redacted && p.Text != "" {
				if reasoningText.Len() > 0 {
					reasoningText.WriteString("\n")
				}
				reasoningText.WriteString(p.Text)
			}
		case model.ToolCallPart:
			toolCalls = append(toolCalls, wireToolCall{
				ID:   p.ToolCallID,
				Type: "function",
				Function: wireFunction{
					Name:      p.ToolName,
					Arguments: string(p.Args),
				},
			})
		case model.ToolResultPart:
			toolResults = append(toolResults, chatMessage{
				Role:       "tool",
				Content:    messageContent{Text: p.Content},
				ToolCallID: p.ToolCallID,
				Name:       p.ToolName,
			})
		case model.ImagePart:
			if msg.Role != model.RoleUser {
				return nil, errors.New("image parts are only supported in user messages")
			}

			url, err := imagePartURL(p)
			if err != nil {
				return nil, err
			}

			content = append(content, contentPart{
				Type: "image_url",
				ImageURL: &contentPartURL{
					URL: url,
				},
			})
		case model.FilePart:
			if msg.Role != model.RoleUser {
				return nil, errors.New("file parts are only supported in user messages")
			}

			filePart, err := userFilePart(p)
			if err != nil {
				return nil, err
			}

			content = append(content, *filePart)
		}
	}

	var result []chatMessage

	switch {
	case len(toolCalls) > 0:
		cm := chatMessage{
			Role:      string(msg.Role),
			ToolCalls: toolCalls,
		}
		if text != "" {
			cm.Content = messageContent{Text: text}
		}
		applyReasoning(&cm, reasoningDetails, reasoningText.String())
		result = append(result, cm)
	case len(content) > 0:
		cm := chatMessage{
			Role:    string(msg.Role),
			Content: condenseContent(content),
		}
		applyReasoning(&cm, reasoningDetails, reasoningText.String())
		result = append(result, cm)
	case msg.Role == model.RoleSystem:
		result = append(result, chatMessage{
			Role:    string(msg.Role),
			Content: messageContent{Text: text},
		})
	case len(reasoningDetails) > 0 || reasoningText.Len() > 0:
		// Pure reasoning assistant turn — keep the block attached to an
		// assistant message so providers that honor it can consume it.
		cm := chatMessage{
			Role: string(msg.Role),
		}
		applyReasoning(&cm, reasoningDetails, reasoningText.String())
		result = append(result, cm)
	}

	result = append(result, toolResults...)

	return result, nil
}

// condenseContent collapses a single-text content list into the plain
// string wire form; arrays with images/audio/files retain the typed
// parts shape.
func condenseContent(parts []contentPart) messageContent {
	if len(parts) == 1 && parts[0].Type == "text" {
		return messageContent{Text: parts[0].Text}
	}

	return messageContent{Parts: parts}
}

// applyReasoning attaches reasoning metadata to an outbound message
// when the caller round-trips a prior reasoning block back to the
// provider.
func applyReasoning(cm *chatMessage, details []wireReasoningPart, text string) {
	if len(details) > 0 {
		cm.ReasoningDetails = details
	}
	if text != "" {
		cm.Reasoning = text
	}
}

// convertTools translates model tool definitions to the wire format.
func convertTools(tools []model.ToolDefinition) []chatTool {
	result := make([]chatTool, len(tools))
	for i, t := range tools {
		result[i] = chatTool{
			Type: "function",
			Function: toolFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.InputSchema,
				Strict:      t.Strict,
			},
		}
	}

	return result
}

func imagePartURL(part model.ImagePart) (string, error) {
	switch {
	case part.URL != "":
		return part.URL, nil
	case len(part.Data) > 0:
		if part.MIMEType == "" {
			return "", errors.New("image data requires mime type")
		}

		return "data:" + part.MIMEType + ";base64," + base64.StdEncoding.EncodeToString(part.Data), nil
	default:
		return "", errors.New("image part requires url or data")
	}
}

// userFilePart converts a user-side FilePart into the appropriate
// OpenAI content part based on MIME type.
func userFilePart(part model.FilePart) (*contentPart, error) {
	switch {
	case strings.HasPrefix(part.MIMEType, "image/"):
		url, err := imagePartURL(model.ImagePart{
			URL:      part.URL,
			Data:     part.Data,
			MIMEType: part.MIMEType,
		})
		if err != nil {
			return nil, err
		}

		return &contentPart{
			Type: "image_url",
			ImageURL: &contentPartURL{
				URL: url,
			},
		}, nil
	case strings.HasPrefix(part.MIMEType, "audio/"):
		if len(part.Data) == 0 {
			return nil, errors.New("audio file parts require inline data")
		}

		return &contentPart{
			Type: "input_audio",
			InputAudio: &contentPartAudio{
				Data:   base64.StdEncoding.EncodeToString(part.Data),
				Format: audioFormatForMIME(part.MIMEType),
			},
		}, nil
	default:
		fp := &contentPartFile{
			Filename: part.Filename,
		}
		switch {
		case len(part.Data) > 0:
			if part.MIMEType == "" {
				return nil, errors.New("file data requires mime type")
			}

			fp.FileData = "data:" + part.MIMEType + ";base64," + base64.StdEncoding.EncodeToString(part.Data)
		case part.URL != "":
			fp.FileID = part.URL
		default:
			return nil, errors.New("file part requires url or data")
		}

		return &contentPart{
			Type: "file",
			File: fp,
		}, nil
	}
}

func audioFormatForMIME(mime string) string {
	switch mime {
	case "audio/wav":
		return "wav"
	case "audio/mpeg", "audio/mp3":
		return "mp3"
	case "audio/flac":
		return "flac"
	case "audio/opus":
		return "opus"
	case "audio/pcm":
		return "pcm16"
	case "audio/aac":
		return "aac"
	default:
		return strings.TrimPrefix(mime, "audio/")
	}
}

func reasoningDetailFromPart(part model.ReasoningPart) (wireReasoningPart, bool) {
	if part.Text == "" && !part.Redacted && part.Signature == "" {
		return wireReasoningPart{}, false
	}

	detail := wireReasoningPart{
		Type:      part.Format,
		ID:        part.ID,
		Signature: part.Signature,
		Format:    part.Format,
	}

	switch {
	case part.Redacted:
		if detail.Type == "" {
			detail.Type = "reasoning.encrypted"
		}
		detail.Data = part.Text
	case detail.Type == "reasoning.summary":
		detail.Summary = part.Text
	default:
		if detail.Type == "" {
			detail.Type = "reasoning.text"
		}
		detail.Text = part.Text
	}

	return detail, true
}

// convertToolChoice translates the model ToolChoice to the wire format.
// OpenAI accepts "auto", "none", "required" as strings, or a specific
// tool as an object.
func convertToolChoice(tc *model.ToolChoice) *toolChoiceValue {
	if tc.Type == "tool" {
		return &toolChoiceValue{
			Named: &toolChoiceSpec{
				Type:     "function",
				Function: toolChoiceSpecFunction{Name: tc.ToolName},
			},
		}
	}

	return &toolChoiceValue{Mode: tc.Type}
}

// convertResponseFormat translates the model ResponseFormat to wire format.
func convertResponseFormat(rf *model.ResponseFormat) *responseFormat {
	r := &responseFormat{Type: rf.Type}

	if rf.JSONSchema != nil {
		r.JSONSchema = &jsonSchemaSpec{
			Name:   rf.JSONSchema.Name,
			Schema: rf.JSONSchema.Schema,
			Strict: rf.JSONSchema.Strict,
		}
	}

	return r
}

// mapFinishReason converts an OpenAI finish reason string to the
// model.FinishReason enum.
func mapFinishReason(reason string) model.FinishReason {
	switch reason {
	case "stop":
		return model.FinishReasonStop
	case "length":
		return model.FinishReasonLength
	case "tool_calls", "function_call":
		return model.FinishReasonToolCalls
	case "content_filter":
		return model.FinishReasonContentFilter
	case "error":
		return model.FinishReasonError
	default:
		return model.FinishReasonUnknown
	}
}

// MarshalRequest serializes a ChatRequest to JSON.
func MarshalRequest(req *ChatRequest) ([]byte, error) {
	return json.Marshal(req)
}
