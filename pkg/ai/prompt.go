package ai

import (
	"fmt"
	"net/http"

	"github.com/aarock1234/ai/pkg/model"
)

// Prompt is the input of a language model call: optional system
// instructions with either a text prompt or a list of messages.
type Prompt struct {
	// System holds instructions for the model. It is sent as a system message
	// before the other messages.
	System string

	// Prompt is a text prompt, sent as a single user message. Set either
	// Prompt or Messages.
	Prompt string

	// Messages is the conversation to continue. Set either Prompt or
	// Messages.
	Messages []model.Message

	// AllowSystemInMessages permits system messages in Messages. They are
	// rejected by default to reduce the risk of prompt injection; prefer
	// System for instructions.
	AllowSystemInMessages bool
}

// CallSettings are the generation settings shared by language model calls.
// Nil pointers and empty values leave the provider default.
type CallSettings struct {
	// MaxOutputTokens limits the number of generated tokens. It must be at
	// least 1.
	MaxOutputTokens *int

	// Temperature controls randomness. Its range depends on the provider.
	// Set either Temperature or TopP, not both.
	Temperature *float64

	// TopP is the nucleus sampling probability mass, between 0 and 1.
	TopP *float64

	// TopK samples only from the K most likely tokens.
	TopK *int

	// PresencePenalty discourages repeating information already in the
	// prompt, from -1 to 1.
	PresencePenalty *float64

	// FrequencyPenalty discourages repeating the same words or phrases,
	// from -1 to 1.
	FrequencyPenalty *float64

	// StopSequences end generation when the model produces one of them.
	StopSequences []string

	// Seed makes sampling deterministic on providers that support it.
	Seed *int

	// Reasoning selects a provider-independent reasoning effort.
	Reasoning model.ReasoningEffort

	// MaxRetries is the number of retries per model call. Nil selects 2;
	// zero disables retries.
	MaxRetries *int

	// Headers are additional HTTP request headers.
	Headers http.Header
}

// standardizedPrompt is a validated prompt: system instructions and a
// non-empty list of messages.
type standardizedPrompt struct {
	instructions []model.SystemMessage
	messages     []model.Message
}

// standardizePrompt validates a Prompt and converts a text prompt into a user
// message.
func standardizePrompt(prompt Prompt) (standardizedPrompt, error) {
	hasPrompt := prompt.Prompt != ""
	hasMessages := prompt.Messages != nil

	if hasPrompt && hasMessages {
		return standardizedPrompt{}, &InvalidPromptError{
			Message: "prompt and messages cannot be defined at the same time",
		}
	}

	if !hasPrompt && !hasMessages {
		return standardizedPrompt{}, &InvalidPromptError{
			Message: "prompt or messages must be defined",
		}
	}

	messages := prompt.Messages
	if hasPrompt {
		messages = []model.Message{
			textUserMessage(prompt.Prompt),
		}
	}

	if len(messages) == 0 {
		return standardizedPrompt{}, &InvalidPromptError{
			Message: "messages must not be empty",
		}
	}

	if err := validateMessages(messages, prompt.AllowSystemInMessages); err != nil {
		return standardizedPrompt{}, err
	}

	return standardizedPrompt{
		instructions: systemInstructions(prompt.System),
		messages:     messages,
	}, nil
}

func textUserMessage(text string) model.UserMessage {
	return model.UserMessage{
		Content: []model.UserPart{
			model.TextPart{
				Text: text,
			},
		},
	}
}

func systemInstructions(system string) []model.SystemMessage {
	if system == "" {
		return nil
	}

	return []model.SystemMessage{
		{
			Content: system,
		},
	}
}

// validateMessages accepts only the four model message types and rejects
// system messages unless they are explicitly allowed.
func validateMessages(messages []model.Message, allowSystem bool) error {
	for index, message := range messages {
		switch message.(type) {
		case model.UserMessage, model.AssistantMessage, model.ToolMessage:
		case model.SystemMessage:
			if !allowSystem {
				return &InvalidPromptError{
					Message: "system messages are not allowed in the prompt or messages fields: use the system option instead",
				}
			}
		case nil:
			return &InvalidPromptError{
				Message: fmt.Sprintf("message %d is nil", index),
			}
		default:
			return &InvalidPromptError{
				Message: fmt.Sprintf("message %d has unsupported type %T", index, message),
				Cause: &InvalidMessageRoleError{
					Role: message.Role(),
				},
			}
		}
	}

	return nil
}

// prepareCallSettings validates the settings and returns the call options
// they produce. The caller adds the prompt, tools, response format and
// provider options.
func prepareCallSettings(settings CallSettings) (model.CallOptions, error) {
	if settings.MaxOutputTokens != nil && *settings.MaxOutputTokens < 1 {
		return model.CallOptions{}, &InvalidArgumentError{
			Parameter: "maxOutputTokens",
			Value:     *settings.MaxOutputTokens,
			Message:   "maxOutputTokens must be >= 1",
		}
	}

	return model.CallOptions{
		MaxOutputTokens:  settings.MaxOutputTokens,
		Temperature:      settings.Temperature,
		StopSequences:    settings.StopSequences,
		TopP:             settings.TopP,
		TopK:             settings.TopK,
		PresencePenalty:  settings.PresencePenalty,
		FrequencyPenalty: settings.FrequencyPenalty,
		Seed:             settings.Seed,
		Reasoning:        settings.Reasoning,
		Headers:          settings.Headers,
	}, nil
}
