package deepagents

import (
	"fmt"
	"strings"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/bridge"
	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
	"github.com/aarock1234/ai/pkg/model"
)

// startCommand starts a turn in the Deep Agents bridge.
type startCommand struct {
	bridge.StartBase

	// Instructions are appended to the Deep Agents system prompt when the
	// bridge builds the session's agent.
	Instructions string `json:"instructions,omitempty"`

	Thinking *Thinking `json:"thinking,omitempty"`
	Effort   Effort    `json:"effort,omitempty"`

	// SkillsPaths are the skills directories of the agent, in increasing
	// precedence.
	SkillsPaths   []string `json:"skillsPaths,omitempty"`
	SkillsChanged bool     `json:"skillsChanged"`

	RecursionLimit int               `json:"recursionLimit,omitempty"`
	MCPServers     map[string]any    `json:"mcpServers,omitzero"`
	Headers        map[string]string `json:"headers,omitempty"`
}

// startCommandOptions configure session.startCommand.
type startCommandOptions struct {
	prompt         string
	settings       harness.TurnSettings
	responseFormat *model.ResponseFormat
	skillsChanged  bool
}

// startCommand builds a turn's start command.
func (s *session) startCommand(options startCommandOptions) startCommand {
	settings := s.harness.settings

	base := bridgeadapter.NewStartBase(bridgeadapter.StartOptions{
		Model:                options.settings.Model,
		Tools:                options.settings.Tools,
		ResponseFormat:       options.responseFormat,
		PermissionMode:       s.permissionMode,
		BuiltinToolFiltering: s.builtinToolFiltering,
	})
	base.Prompt = options.prompt

	return startCommand{
		StartBase:      base,
		Instructions:   options.settings.Instructions,
		Thinking:       settings.Thinking,
		Effort:         settings.Effort,
		SkillsPaths:    s.skillsPaths,
		SkillsChanged:  options.skillsChanged,
		RecursionLimit: settings.RecursionLimit,
		MCPServers:     settings.MCPServers,
		Headers:        s.headers,
	}
}

// promptText reduces a prompt to its text, joining text parts with blank
// lines. Other parts are unsupported.
func promptText(prompt model.UserMessage) (string, error) {
	texts := make([]string, 0, len(prompt.Content))

	for _, part := range prompt.Content {
		text, ok := part.(model.TextPart)
		if !ok {
			return "", &harness.CapabilityUnsupportedError{
				Message: fmt.Sprintf(
					"The deepagents harness does not yet support user message parts of type '%s'. Pass a string or a user message whose content contains only text parts.",
					userPartType(part),
				),
				HarnessID: HarnessID,
			}
		}

		texts = append(texts, text.Text)
	}

	return strings.Join(texts, "\n\n"), nil
}

// userPartType names a user message part as upstream's part type.
func userPartType(part model.UserPart) string {
	if _, ok := part.(model.FilePart); ok {
		return "file"
	}

	return fmt.Sprintf("%T", part)
}
