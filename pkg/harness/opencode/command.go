package opencode

import (
	"fmt"
	"strings"

	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/bridge"
	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
	"github.com/aarock1234/ai/pkg/model"
)

// operation is what a start command asks the bridge to do.
type operation string

// Operations.
const (
	operationPrompt  operation = "prompt"
	operationCompact operation = "compact"
)

// startCommand starts an operation in the OpenCode bridge. The bridge
// disables built-in tools through InactiveBuiltinToolNames instead of the
// shared filtering field.
type startCommand struct {
	bridge.StartBase

	Operation    operation `json:"operation"`
	Provider     string    `json:"provider,omitempty"`
	Variant      string    `json:"variant,omitempty"`
	Instructions string    `json:"instructions,omitempty"`

	// SkillsChanged makes OpenCode reload the skills directory.
	SkillsChanged bool `json:"skillsChanged,omitempty"`

	// ResumeSessionID makes the bridge continue the OpenCode session instead
	// of creating one.
	ResumeSessionID string `json:"resumeSessionId,omitempty"`

	OpenCodeConfig map[string]any    `json:"openCodeConfig,omitempty"`
	MCPServers     map[string]any    `json:"mcpServers,omitempty"`
	Headers        map[string]string `json:"headers,omitempty"`

	// InactiveBuiltinToolNames is nil without filtering and empty when the
	// filtering keeps every tool.
	InactiveBuiltinToolNames []string `json:"inactiveBuiltinToolNames,omitzero"`
}

// promptCommandOptions configure session.promptCommand.
type promptCommandOptions struct {
	prompt         string
	settings       harness.TurnSettings
	responseFormat *model.ResponseFormat
	model          string
	skillsChanged  bool
	resumeID       string
}

// promptCommand builds the start command of a prompt turn.
func (s *session) promptCommand(options promptCommandOptions) startCommand {
	command := s.baseCommand(baseCommandOptions{
		operation:      operationPrompt,
		prompt:         options.prompt,
		tools:          options.settings.Tools,
		responseFormat: options.responseFormat,
		model:          options.model,
		resumeID:       options.resumeID,
	})

	command.Variant = s.harness.settings.ReasoningVariant
	command.Instructions = options.settings.Instructions
	command.SkillsChanged = options.skillsChanged

	return command
}

// baseCommandOptions configure session.baseCommand.
type baseCommandOptions struct {
	operation      operation
	prompt         string
	tools          []harness.ToolSpec
	responseFormat *model.ResponseFormat
	model          string
	resumeID       string
}

// baseCommand builds the fields that prompt and compaction commands share.
func (s *session) baseCommand(options baseCommandOptions) startCommand {
	base := bridgeadapter.NewStartBase(bridgeadapter.StartOptions{
		Model:          options.model,
		Tools:          options.tools,
		ResponseFormat: options.responseFormat,
		PermissionMode: s.permissionMode,
		Observability:  s.observability,
	})
	base.Prompt = options.prompt

	return startCommand{
		StartBase:                base,
		Operation:                options.operation,
		Provider:                 s.harness.settings.Provider,
		ResumeSessionID:          options.resumeID,
		OpenCodeConfig:           s.openCodeConfig,
		MCPServers:               s.harness.settings.MCPServers,
		Headers:                  s.headers,
		InactiveBuiltinToolNames: s.inactiveBuiltinToolNames,
	}
}

// promptText returns the text of a prompt, joining its parts with blank
// lines. OpenCode turns accept text only.
func promptText(prompt model.UserMessage) (string, error) {
	texts := make([]string, 0, len(prompt.Content))

	for _, part := range prompt.Content {
		text, ok := part.(model.TextPart)
		if !ok {
			return "", unsupported(fmt.Sprintf(
				"The OpenCode harness does not yet support user message parts of type '%s'. Pass a string or a user message whose content contains only text parts.",
				partType(part),
			))
		}

		texts = append(texts, text.Text)
	}

	return strings.Join(texts, "\n\n"), nil
}

// partType names a prompt part as upstream does: images are file parts of an
// image media type.
func partType(part model.UserPart) string {
	file, ok := part.(model.FilePart)
	if !ok {
		return fmt.Sprintf("%T", part)
	}

	if mediatype.TopLevel(file.MediaType) == "image" {
		return "image"
	}

	return "file"
}
