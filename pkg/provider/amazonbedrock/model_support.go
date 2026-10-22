package amazonbedrock

import (
	"regexp"
	"strings"
	"unicode/utf16"
)

// ModelFamily names the model family behind a model ID that does not reveal
// it, such as an application inference profile ARN.
type ModelFamily string

// Model families.
const (
	// ModelFamilyAnthropic marks a Claude chat model.
	ModelFamilyAnthropic ModelFamily = "anthropic"

	// ModelFamilyTitan, ModelFamilyCohere and ModelFamilyNova mark
	// embedding models.
	ModelFamilyTitan  ModelFamily = "titan"
	ModelFamilyCohere ModelFamily = "cohere"
	ModelFamilyNova   ModelFamily = "nova"
)

var (
	// openAIModelPattern captures the OpenAI model of an ID with an
	// optional region prefix.
	openAIModelPattern = regexp.MustCompile(`^(?:[^.]+\.)?(openai\..+)$`)

	// mistralToolCallIDPattern matches tool call IDs that Mistral accepts.
	mistralToolCallIDPattern = regexp.MustCompile(`^[a-zA-Z0-9]{9}$`)
)

// isAnthropicModel reports whether a chat model is Claude: by family, by
// ID, or by an inference profile used with a thinking budget.
func isAnthropicModel(modelID string, family ModelFamily, reasoningBudgetTokens *int) bool {
	return family == ModelFamilyAnthropic ||
		strings.Contains(modelID, "anthropic") ||
		(strings.Contains(modelID, ":application-inference-profile/") && reasoningBudgetTokens != nil)
}

// openAIModel returns the OpenAI model name of an OpenAI model ID.
func openAIModel(modelID string) (string, bool) {
	match := openAIModelPattern.FindStringSubmatch(modelID)
	if match == nil {
		return "", false
	}

	return match[1], true
}

func isNova2ReasoningModel(modelID string) bool {
	return strings.Contains(modelID, "amazon.nova-2-lite-v1:0")
}

// isMistralModel reports whether a model is Mistral, with or without a
// region prefix.
func isMistralModel(modelID string) bool {
	return strings.Contains(modelID, "mistral.")
}

const (
	base62Characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

	normalizedToolCallIDLength = 9

	fnvOffsetBasis64 uint64 = 14695981039346656037
	fnvPrime64       uint64 = 1099511628211
)

// normalizeToolCallID returns an ID Mistral accepts, nine alphanumeric
// characters, for Mistral models. Other IDs are hashed deterministically;
// other models keep their IDs.
func normalizeToolCallID(toolCallID string, mistral bool) string {
	if !mistral || mistralToolCallIDPattern.MatchString(toolCallID) {
		return toolCallID
	}

	return base62Hash(toolCallID)
}

// base62Hash hashes the UTF-16 code units of value with 64-bit FNV-1a, as
// upstream does in JavaScript, and encodes the hash as nine base62 digits.
func base62Hash(value string) string {
	hash := fnvOffsetBasis64
	for _, unit := range utf16.Encode([]rune(value)) {
		hash ^= uint64(unit)
		hash *= fnvPrime64
	}

	base := uint64(len(base62Characters))

	digitSpace := uint64(1)
	for range normalizedToolCallIDLength {
		digitSpace *= base
	}

	remaining := hash % digitSpace
	digits := make([]byte, normalizedToolCallIDLength)

	for index := normalizedToolCallIDLength - 1; index >= 0; index-- {
		digits[index] = base62Characters[remaining%base]
		remaining /= base
	}

	return string(digits)
}
