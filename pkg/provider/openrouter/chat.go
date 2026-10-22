package openrouter

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openaicompatible"
)

// ChatModel is an OpenRouter chat model. It sends OpenAI-compatible chat
// completions and adds OpenRouter's response extensions: reasoning details,
// generated images and web citations.
type ChatModel struct {
	provider   string
	chat       *openaicompatible.ChatModel
	generateID func() string
}

// Provider returns the provider identifier, "openrouter.chat" by default.
func (m *ChatModel) Provider() string {
	return m.provider
}

// ModelID returns the OpenRouter model identifier.
func (m *ChatModel) ModelID() string {
	return m.chat.ModelID()
}

// SupportedURLs reports that OpenRouter downloads image URLs itself.
func (m *ChatModel) SupportedURLs() map[string][]*regexp.Regexp {
	return m.chat.SupportedURLs()
}

// Generate performs a non-streaming chat completion.
func (m *ChatModel) Generate(ctx context.Context, options model.CallOptions) (*model.GenerateResult, error) {
	options, err := prepareCall(options)
	if err != nil {
		return nil, err
	}

	result, err := m.chat.Generate(ctx, options)
	if err != nil {
		return nil, err
	}

	var extensions responseExtensions
	if err := json.Unmarshal(result.Response.Body, &extensions); err != nil {
		return nil, fmt.Errorf("%w: %w", model.ErrInvalidResponse, err)
	}

	content, err := extendContent(result.Content, extensions.first(), m.generateID)
	if err != nil {
		return nil, err
	}

	result.Content = content

	return result, nil
}

// Stream performs a streaming chat completion.
func (m *ChatModel) Stream(ctx context.Context, options model.CallOptions) (*model.StreamResult, error) {
	options, err := prepareCall(options)
	if err != nil {
		return nil, err
	}

	// The extensions are read from the raw chunks, which are passed on only
	// when the caller asked for them.
	includeRawChunks := options.IncludeRawChunks
	options.IncludeRawChunks = true

	inner, err := m.chat.Stream(ctx, options)
	if err != nil {
		return nil, err
	}

	stream := &extensionStream{
		includeRawChunks: includeRawChunks,
		generateID:       m.generateID,
	}

	result := model.NewStreamResult(stream.parts(inner.Parts()), inner.Close)
	result.Request = inner.Request
	result.Response = inner.Response

	return result, nil
}

// prepareCall validates the OpenRouter options and moves replayed reasoning
// details onto their assistant messages.
func prepareCall(options model.CallOptions) (model.CallOptions, error) {
	if _, err := readChatOptions(options.ProviderOptions); err != nil {
		return model.CallOptions{}, err
	}

	prompt, err := replayReasoningDetails(options.Prompt)
	if err != nil {
		return model.CallOptions{}, err
	}

	options.Prompt = prompt

	return options, nil
}

// extendContent attaches the reasoning details to the reasoning and appends
// generated images and citation sources.
func extendContent(content []model.Content, extensions messageExtensions, generateID func() string) ([]model.Content, error) {
	extended := attachReasoningDetails(content, extensions.ReasoningDetails)

	files, err := imageFiles(extensions.Images)
	if err != nil {
		return nil, err
	}

	for _, file := range files {
		extended = append(extended, file)
	}

	for _, source := range citationSources(extensions.Annotations, generateID) {
		extended = append(extended, source)
	}

	return extended, nil
}

// attachReasoningDetails stores the details on the first reasoning content.
// Reasoning that has no text, such as encrypted reasoning, gets an empty
// reasoning item so that it can be replayed.
func attachReasoningDetails(content []model.Content, details []ReasoningDetail) []model.Content {
	if len(details) == 0 {
		return content
	}

	for index, item := range content {
		reasoning, ok := item.(model.Reasoning)
		if !ok {
			continue
		}

		reasoning.ProviderMetadata = reasoningMetadata(details)
		content[index] = reasoning

		return content
	}

	reasoning := model.Reasoning{
		ProviderMetadata: reasoningMetadata(details),
	}

	return append([]model.Content{reasoning}, content...)
}

var _ model.LanguageModel = (*ChatModel)(nil)
