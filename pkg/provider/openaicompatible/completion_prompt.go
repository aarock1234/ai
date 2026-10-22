package openaicompatible

import (
	"fmt"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

// Speaker labels of the completion transcript.
const (
	userLabel      = "user"
	assistantLabel = "assistant"
)

// completionPrompt is a chat prompt rendered as a transcript.
type completionPrompt struct {
	Text string

	// StopSequences end the completion when the model starts a user turn.
	StopSequences []string
}

// convertCompletionPrompt renders the prompt as "user:" and "assistant:"
// turns, ending with an open assistant turn. A system message is accepted
// only at the start, and tool content is rejected.
func convertCompletionPrompt(prompt []model.Message) (completionPrompt, error) {
	var text strings.Builder

	if len(prompt) > 0 {
		if system, ok := prompt[0].(model.SystemMessage); ok {
			writeParagraph(&text, system.Content)
			prompt = prompt[1:]
		}
	}

	for _, message := range prompt {
		switch message := message.(type) {
		case model.SystemMessage:
			return completionPrompt{}, fmt.Errorf("%w: unexpected system message in prompt: %s", model.ErrInvalidPrompt, message.Content)

		case model.UserMessage:
			writeTurn(&text, userLabel, userText(message))

		case model.AssistantMessage:
			assistantText, err := assistantText(message)
			if err != nil {
				return completionPrompt{}, err
			}

			writeTurn(&text, assistantLabel, assistantText)

		case model.ToolMessage:
			return completionPrompt{}, fmt.Errorf("%w: tool messages", model.ErrUnsupported)

		default:
			return completionPrompt{}, fmt.Errorf("%w: unsupported message role %q", model.ErrInvalidPrompt, message.Role())
		}
	}

	text.WriteString(assistantLabel + ":\n")

	return completionPrompt{
		Text: text.String(),
		StopSequences: []string{
			"\n" + userLabel + ":",
		},
	}, nil
}

// writeTurn writes one labeled turn of the transcript: the label on its own
// line, the content, and a blank line.
func writeTurn(text *strings.Builder, label string, content string) {
	text.WriteString(label)
	text.WriteString(":\n")

	writeParagraph(text, content)
}

// writeParagraph writes content followed by a blank line.
func writeParagraph(text *strings.Builder, content string) {
	text.WriteString(content)
	text.WriteString("\n\n")
}

// userText joins the text parts. Files are not sent.
func userText(message model.UserMessage) string {
	var text strings.Builder
	for _, part := range message.Content {
		if part, ok := part.(model.TextPart); ok {
			text.WriteString(part.Text)
		}
	}

	return text.String()
}

// assistantText joins the text parts. Tool calls are rejected and other parts
// are not sent.
func assistantText(message model.AssistantMessage) (string, error) {
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
