package otel

import (
	"encoding/json"
	"math"
	"regexp"
	"strings"

	"go.opentelemetry.io/otel/attribute"

	"github.com/aarock1234/ai/pkg/ai"
)

// setTokenUsage records the semantic conventions' token counts.
func setTokenUsage(attributes *spanAttributes, usage ai.LanguageModelUsage) {
	attributes.setInt("gen_ai.usage.input_tokens", usage.InputTokens)
	attributes.setInt("gen_ai.usage.output_tokens", usage.OutputTokens)
	attributes.setInt("gen_ai.usage.cache_read.input_tokens", usage.InputTokenDetails.CacheReadTokens)
	attributes.setInt("gen_ai.usage.cache_creation.input_tokens", usage.InputTokenDetails.CacheWriteTokens)
}

// setDetailedUsage records the AI SDK usage details that the semantic
// conventions do not define.
func setDetailedUsage(attributes *spanAttributes, usage ai.LanguageModelUsage) {
	attributes.setInt("ai.usage.inputTokenDetails.noCacheTokens", usage.InputTokenDetails.NoCacheTokens)
	attributes.setInt("ai.usage.outputTokenDetails.textTokens", usage.OutputTokenDetails.TextTokens)
	attributes.setInt("ai.usage.outputTokenDetails.reasoningTokens", usage.OutputTokenDetails.ReasoningTokens)
}

// usageKeyAliases map normalized provider usage keys to common names.
var usageKeyAliases = map[string]string{
	"inputtokens":          "input_tokens",
	"prompttokens":         "input_tokens",
	"prompttokencount":     "input_tokens",
	"totalinputtokens":     "input_tokens",
	"outputtokens":         "output_tokens",
	"completiontokens":     "output_tokens",
	"completiontokencount": "output_tokens",
	"candidatestokencount": "output_tokens",
	"totaltokens":          "total_tokens",
	"totaltokencount":      "total_tokens",
}

var (
	nonAlphanumeric    = regexp.MustCompile(`[^a-zA-Z0-9]`)
	nonAlphanumericRun = regexp.MustCompile(`[^a-zA-Z0-9]+`)
	camelCaseBoundary  = regexp.MustCompile(`([a-z0-9])([A-Z])`)
)

// providerUsageAttributes converts a provider's native usage object into
// numeric attributes below prefix, one per number, keyed by the snake case
// path to it. Units such as tokens, characters and seconds stay distinct so
// that backends can price them.
func providerUsageAttributes(usage json.RawMessage, prefix string) []attribute.KeyValue {
	if len(usage) == 0 {
		return nil
	}

	var decoded any
	if err := json.Unmarshal(usage, &decoded); err != nil {
		return nil
	}

	var attributes []attribute.KeyValue
	addUsageValue(&attributes, prefix, decoded, nil)

	return attributes
}

func addUsageValue(attributes *[]attribute.KeyValue, prefix string, value any, path []string) {
	switch typed := value.(type) {
	case float64:
		if len(path) == 0 || math.IsInf(typed, 0) || math.IsNaN(typed) {
			return
		}

		key := prefix + "." + strings.Join(path, ".")
		*attributes = append(*attributes, attribute.Float64(key, typed))

	case map[string]any:
		for key, nested := range typed {
			normalized := normalizeUsageKey(key)
			if normalized == "" {
				continue
			}

			addUsageValue(attributes, prefix, nested, append(path[:len(path):len(path)], normalized))
		}
	}
}

func normalizeUsageKey(key string) string {
	aliasKey := strings.ToLower(nonAlphanumeric.ReplaceAllString(key, ""))
	if alias, ok := usageKeyAliases[aliasKey]; ok {
		return alias
	}

	normalized := camelCaseBoundary.ReplaceAllString(key, "${1}_${2}")
	normalized = nonAlphanumericRun.ReplaceAllString(normalized, "_")
	normalized = strings.Trim(normalized, "_")

	return strings.ToLower(normalized)
}
