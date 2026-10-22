package minimax

import (
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/anthropic"
)

// ThinkingType selects adaptive thinking or turns it off.
type ThinkingType string

// Thinking types.
const (
	ThinkingAdaptive ThinkingType = "adaptive"
	ThinkingDisabled ThinkingType = "disabled"
)

// ChatOptions are the MiniMax-specific provider options of chat models, read
// from the "minimax" namespace. The namespace also accepts every
// anthropic.MessagesOptions field, which overrides the same field under the
// "anthropic" namespace.
type ChatOptions struct {
	// Effort is the reasoning effort of MiniMax M3.1 and later models. Higher
	// levels can use more reasoning tokens and increase latency. Empty uses
	// the server's setting.
	Effort anthropic.Effort `json:"effort,omitempty"`

	// Thinking turns adaptive thinking on or off.
	Thinking *Thinking `json:"thinking,omitempty"`
}

// Thinking configures thinking.
type Thinking struct {
	Type ThinkingType `json:"type"`
}

// CacheControlOptions mark a stable prompt prefix of an M3.1 or later model
// for caching. Set them under the "minimax" namespace of system messages and
// text parts. A cache hit is not guaranteed.
type CacheControlOptions = anthropic.CacheControlOptions

// readChatOptions decodes the "minimax" provider options. The anthropic
// package validates them, together with the rest of the namespace, when the
// request is prepared.
func readChatOptions(options model.ProviderOptions) (ChatOptions, error) {
	chatOptions, _, err := model.ProviderValue[ChatOptions](options, Namespace)
	if err != nil {
		return ChatOptions{}, fmt.Errorf("minimax provider options: %w", err)
	}

	return chatOptions, nil
}

// anthropicEffort reports whether the "anthropic" namespace sets an effort.
func anthropicEffort(options model.ProviderOptions) (bool, error) {
	messagesOptions, _, err := model.ProviderValue[anthropic.MessagesOptions](options, "anthropic")
	if err != nil {
		return false, fmt.Errorf("anthropic provider options: %w", err)
	}

	return messagesOptions.Effort != "", nil
}
