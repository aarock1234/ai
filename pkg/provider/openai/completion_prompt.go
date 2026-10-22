package openai

import (
	"fmt"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

const (
	completionUserLabel      = "user"
	completionAssistantLabel = "assistant"
)

// completionPrompt renders a conversation as a plain text transcript that
// ends with an open assistant turn. A leading system message becomes the
// preamble. The returned stop sequence ends generation at the next user turn.
func completionPrompt(prompt []model.Message) (string, []string, error) {
	var text strings.Builder

	if len(prompt) > 0 {
		if system, ok := prompt[0].(model.SystemMessage); ok {
			text.WriteString(system.Content)
			text.WriteString("\n\n")
			prompt = prompt[1:]
		}
	}

	for _, message := range prompt {
		turn, err := completionTurn(message)
		if err != nil {
			return "", nil, err
		}

		text.WriteString(turn)
	}

	text.WriteString(completionAssistantLabel + ":\n")

	stopSequences := []string{
		"\n" + completionUserLabel + ":",
	}

	return text.String(), stopSequences, nil
}

func completionTurn(message model.Message) (string, error) {
	switch message := message.(type) {
	case model.SystemMessage:
		return "", fmt.Errorf("%w: unexpected system message in prompt: %s", model.ErrInvalidPrompt, message.Content)

	case model.UserMessage:
		return completionUserLabel + ":\n" + completionUserText(message) + "\n\n", nil

	case model.AssistantMessage:
		text, err := completionAssistantText(message)
		if err != nil {
			return "", err
		}

		return completionAssistantLabel + ":\n" + text + "\n\n", nil

	case model.ToolMessage:
		return "", fmt.Errorf("%w: tool messages", model.ErrUnsupported)

	default:
		return "", fmt.Errorf("%w: unsupported message role %q", model.ErrInvalidPrompt, message.Role())
	}
}

// completionUserText joins the text parts of a user message; files have no
// text form and are left out.
func completionUserText(message model.UserMessage) string {
	var text strings.Builder

	for _, part := range message.Content {
		if textPart, ok := part.(model.TextPart); ok {
			text.WriteString(textPart.Text)
		}
	}

	return text.String()
}

// completionAssistantText joins the text parts of an assistant message. Tool
// calls cannot be represented in a transcript.
func completionAssistantText(message model.AssistantMessage) (string, error) {
	var text strings.Builder

	for _, part := range message.Content {
		switch part := part.(type) {
		case model.TextPart:
			text.WriteString(part.Text)

		case model.ToolCallPart:
			return "", fmt.Errorf("%w: tool-call messages", model.ErrUnsupported)
		}
	}

	return text.String(), nil
}
