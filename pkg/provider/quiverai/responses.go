package quiverai

import (
	"context"
	"iter"
	"maps"
	"regexp"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openresponses"
)

// ResponsesModel is a QuiverAI language model served by the Open Responses
// API. It applies QuiverAI's request policy: only QuiverAI reasoning efforts
// and the "auto" summary are accepted, reasoning "none" is omitted, and
// replayed reasoning keeps only its item ID and summary.
type ResponsesModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "quiverai.responses" by default.
func (m *ResponsesModel) Provider() string {
	return m.provider.name + ".responses"
}

// ModelID returns the model identifier.
func (m *ResponsesModel) ModelID() string {
	return m.modelID
}

// SupportedURLs reports the URLs that the Open Responses endpoint downloads
// itself.
func (m *ResponsesModel) SupportedURLs() map[string][]*regexp.Regexp {
	return m.provider.responsesProvider("").Responses(m.modelID).SupportedURLs()
}

// Generate performs a non-streaming Responses API call.
func (m *ResponsesModel) Generate(ctx context.Context, options model.CallOptions) (*model.GenerateResult, error) {
	options, warnings, err := prepareCall(options)
	if err != nil {
		return nil, err
	}

	inner, err := m.inner()
	if err != nil {
		return nil, err
	}

	result, err := inner.Generate(ctx, options)
	if err != nil {
		return nil, err
	}

	result.Warnings = slices.Concat(warnings, result.Warnings)

	return result, nil
}

// Stream performs a streaming Responses API call.
func (m *ResponsesModel) Stream(ctx context.Context, options model.CallOptions) (*model.StreamResult, error) {
	options, warnings, err := prepareCall(options)
	if err != nil {
		return nil, err
	}

	inner, err := m.inner()
	if err != nil {
		return nil, err
	}

	stream, err := inner.Stream(ctx, options)
	if err != nil {
		return nil, err
	}

	if len(warnings) == 0 {
		return stream, nil
	}

	result := model.NewStreamResult(withStartWarnings(stream.Parts(), warnings), stream.Close)
	result.Request = stream.Request
	result.Response = stream.Response

	return result, nil
}

// inner returns the Open Responses model of one call.
func (m *ResponsesModel) inner() (*openresponses.ResponsesModel, error) {
	apiKey, err := m.provider.loadAPIKey()
	if err != nil {
		return nil, err
	}

	return m.provider.responsesProvider(apiKey).Responses(m.modelID), nil
}

// prepareCall validates the QuiverAI options, omits reasoning "none" unless
// an effort is set, and reduces replayed reasoning to what QuiverAI accepts.
func prepareCall(options model.CallOptions) (model.CallOptions, []model.Warning, error) {
	responsesOptions, err := readResponsesOptions(options.ProviderOptions)
	if err != nil {
		return model.CallOptions{}, nil, err
	}

	var warnings []model.Warning

	if options.Reasoning == model.ReasoningNone && responsesOptions.ReasoningEffort == nil {
		warnings = append(warnings, model.UnsupportedWarning("reasoning effort none", ""))
		options.Reasoning = ""
	}

	options.Prompt = withReplayableReasoning(options.Prompt)

	return options, warnings, nil
}

// withReplayableReasoning returns a copy of the prompt whose assistant
// reasoning parts have no text and only the item ID and summary of their
// QuiverAI metadata.
func withReplayableReasoning(prompt []model.Message) []model.Message {
	converted := make([]model.Message, 0, len(prompt))

	for _, message := range prompt {
		assistant, ok := message.(model.AssistantMessage)
		if !ok {
			converted = append(converted, message)

			continue
		}

		content := make([]model.AssistantPart, 0, len(assistant.Content))
		for _, part := range assistant.Content {
			if reasoning, ok := part.(model.ReasoningPart); ok {
				part = replayableReasoning(reasoning)
			}

			content = append(content, part)
		}

		assistant.Content = content
		converted = append(converted, assistant)
	}

	return converted
}

func replayableReasoning(part model.ReasoningPart) model.ReasoningPart {
	providerOptions := maps.Clone(part.ProviderOptions)
	if providerOptions == nil {
		providerOptions = make(model.ProviderOptions)
	}

	providerOptions[Namespace] = readReasoningReplay(part.ProviderOptions)

	return model.ReasoningPart{
		Text:            "",
		ProviderOptions: providerOptions,
	}
}

// withStartWarnings adds the call warnings before the stream start
// warnings.
func withStartWarnings(inner iter.Seq[model.StreamPart], warnings []model.Warning) iter.Seq[model.StreamPart] {
	return func(yield func(model.StreamPart) bool) {
		for part := range inner {
			if start, ok := part.(model.StreamStart); ok {
				start.Warnings = slices.Concat(warnings, start.Warnings)
				part = start
			}

			if !yield(part) {
				return
			}
		}
	}
}

var _ model.LanguageModel = (*ResponsesModel)(nil)
