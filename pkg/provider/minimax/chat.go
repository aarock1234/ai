package minimax

import (
	"context"
	"fmt"
	"iter"
	"maps"
	"regexp"
	"strconv"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/internal/reasoning"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/anthropic"
)

// modelVersion matches the major and minor version of M-series model IDs.
// Later M-series models are assumed to keep the M3.1 capabilities.
var modelVersion = regexp.MustCompile(`(?i)^minimax-m(\d+)(?:\.(\d+))?(?:\.\d+)*(?:-|$)`)

// m31SupportedURLs are the image and video URLs that M3.1 and later models
// download themselves.
var m31SupportedURLs = map[string][]*regexp.Regexp{
	"image/*": {
		regexp.MustCompile(`^https?://`),
	},
	"video/*": {
		regexp.MustCompile(`^https?://`),
	},
}

// m31MaxTemperature is the maximum temperature of M3.1 and later models.
const m31MaxTemperature = 2.0

// effortMap maps reasoning levels to the efforts of M3.1 and later models.
var effortMap = map[model.ReasoningEffort]anthropic.Effort{
	model.ReasoningMinimal: anthropic.EffortLow,
	model.ReasoningLow:     anthropic.EffortLow,
	model.ReasoningMedium:  anthropic.EffortMedium,
	model.ReasoningHigh:    anthropic.EffortHigh,
	model.ReasoningXHigh:   anthropic.EffortXHigh,
	model.ReasoningMax:     anthropic.EffortMax,
}

// ChatModel is a MiniMax chat model served by the Anthropic-compatible
// Messages API. On M3.1 and later models, the provider-independent
// reasoning level selects an effort instead of a thinking budget, image and
// video parts accept ImagePartOptions and VideoPartOptions, temperatures up
// to 2 are accepted, and temperature and topP are kept with thinking.
type ChatModel struct {
	modelID             string
	provider            *Provider
	supportsM31Features bool
}

// Provider returns the provider identifier, "minimax.messages" by default.
func (m *ChatModel) Provider() string {
	return m.provider.name + ".messages"
}

// ModelID returns the model identifier.
func (m *ChatModel) ModelID() string {
	return m.modelID
}

// SupportedURLs reports that M3.1 and later models download image and video
// URLs themselves. Earlier models receive every file inline.
func (m *ChatModel) SupportedURLs() map[string][]*regexp.Regexp {
	if m.supportsM31Features {
		return m31SupportedURLs
	}

	return map[string][]*regexp.Regexp{}
}

// Generate performs one non-streaming Messages API call.
func (m *ChatModel) Generate(ctx context.Context, options model.CallOptions) (*model.GenerateResult, error) {
	options, warnings, err := m.prepareCall(options)
	if err != nil {
		return nil, err
	}

	messages, err := m.provider.messages(m.messagesOptions()...)
	if err != nil {
		return nil, err
	}

	result, err := messages.Messages(m.modelID).Generate(ctx, options)
	if err != nil {
		return nil, err
	}

	result.Warnings = append(result.Warnings, warnings...)

	return result, nil
}

// Stream performs one streaming Messages API call.
func (m *ChatModel) Stream(ctx context.Context, options model.CallOptions) (*model.StreamResult, error) {
	options, warnings, err := m.prepareCall(options)
	if err != nil {
		return nil, err
	}

	messages, err := m.provider.messages(m.messagesOptions()...)
	if err != nil {
		return nil, err
	}

	inner, err := messages.Messages(m.modelID).Stream(ctx, options)
	if err != nil {
		return nil, err
	}

	result := model.NewStreamResult(withStartWarnings(inner.Parts(), warnings), inner.Close)
	result.Request = inner.Request
	result.Response = inner.Response

	return result, nil
}

// messagesOptions returns the Anthropic options of M3.1 and later models.
func (m *ChatModel) messagesOptions() []anthropic.Option {
	if !m.supportsM31Features {
		return nil
	}

	return []anthropic.Option{
		anthropic.WithMediaPartConverter(convertMediaPart),
		anthropic.WithMaxTemperature(m31MaxTemperature),
		anthropic.WithTemperatureAndTopPWithThinking(true),
	}
}

// prepareCall replaces the reasoning level of M3.1 and later models with an
// effort, unless the call already sets one.
func (m *ChatModel) prepareCall(options model.CallOptions) (model.CallOptions, []model.Warning, error) {
	if !m.supportsM31Features || !reasoning.IsCustom(options.Reasoning) {
		return options, nil, nil
	}

	chatOptions, err := readChatOptions(options.ProviderOptions)
	if err != nil {
		return model.CallOptions{}, nil, err
	}

	anthropicSetsEffort, err := anthropicEffort(options.ProviderOptions)
	if err != nil {
		return model.CallOptions{}, nil, err
	}

	if chatOptions.Effort != "" || anthropicSetsEffort {
		return options, nil, nil
	}

	effort, warnings, ok := m.reasoningEffort(options.Reasoning)

	// M3.1 and later use effort instead of Anthropic thinking budgets.
	options.Reasoning = ""

	if !ok {
		return options, warnings, nil
	}

	providerOptions, err := withEffort(options.ProviderOptions, effort)
	if err != nil {
		return model.CallOptions{}, nil, err
	}

	options.ProviderOptions = providerOptions

	return options, warnings, nil
}

// reasoningEffort maps a reasoning level to an effort. These models always
// reason, so "none" uses the lowest effort.
func (m *ChatModel) reasoningEffort(level model.ReasoningEffort) (anthropic.Effort, []model.Warning, bool) {
	if level != model.ReasoningNone {
		return reasoning.Effort(level, effortMap)
	}

	details := fmt.Sprintf(
		"reasoning 'none' is not supported by %s; it always reasons. Using effort 'low' to minimize thinking instead.",
		m.modelID,
	)

	return anthropic.EffortLow, []model.Warning{
		model.CompatibilityWarning("reasoning", details),
	}, true
}

// withEffort returns a copy of the provider options whose "minimax"
// namespace sets effort, keeping the namespace's other members.
func withEffort(options model.ProviderOptions, effort anthropic.Effort) (model.ProviderOptions, error) {
	namespace, _, err := model.ProviderValue[jsonobject.Object](options, Namespace)
	if err != nil {
		return nil, fmt.Errorf("minimax provider options: %w", err)
	}

	updated := maps.Clone(options)
	if updated == nil {
		updated = make(model.ProviderOptions)
	}

	updated[Namespace] = namespace.With("effort", jsonobject.EncodeString(string(effort)))

	return updated, nil
}

// withStartWarnings adds the call warnings to the stream start.
func withStartWarnings(inner iter.Seq[model.StreamPart], warnings []model.Warning) iter.Seq[model.StreamPart] {
	return func(yield func(model.StreamPart) bool) {
		for part := range inner {
			if start, ok := part.(model.StreamStart); ok {
				start.Warnings = append(start.Warnings, warnings...)
				part = start
			}

			if !yield(part) {
				return
			}
		}
	}
}

// supportsM31Features reports whether a model ID names MiniMax M3.1 or a
// later M-series model. Version components compare as integers, so M3.123
// is newer than M3.1.
func supportsM31Features(modelID string) bool {
	match := modelVersion.FindStringSubmatch(modelID)
	if match == nil {
		return false
	}

	major, err := strconv.Atoi(match[1])
	if err != nil {
		return false
	}

	minor := 0
	if match[2] != "" {
		minor, err = strconv.Atoi(match[2])
		if err != nil {
			return false
		}
	}

	return major > 3 || (major == 3 && minor >= 1)
}

var _ model.LanguageModel = (*ChatModel)(nil)
