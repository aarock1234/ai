package claudecode

import (
	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/bridge"
	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
	"github.com/aarock1234/ai/pkg/model"
)

// startCommand starts a turn in the Claude Code bridge. Its Prompt replaces
// the text prompt of the shared start fields.
type startCommand struct {
	bridge.StartBase

	Prompt       harness.PromptContent `json:"prompt"`
	Instructions string                `json:"instructions,omitempty"`

	Thinking               Thinking          `json:"thinking"`
	Effort                 Effort            `json:"effort,omitempty"`
	MaxTurns               int               `json:"maxTurns,omitempty"`
	AgentProgressSummaries *bool             `json:"agentProgressSummaries,omitempty"`
	ForwardSubagentText    *bool             `json:"forwardSubagentText,omitempty"`
	Env                    map[string]string `json:"env,omitempty"`
	Skills                 []string          `json:"skills,omitempty"`
	MCPServers             map[string]any    `json:"mcpServers,omitempty"`

	resumeFields
}

// resumeFields select the conversation a new SDK query resumes. The SDK
// accepts one or the other.
type resumeFields struct {
	// Continue rehydrates the working directory's most recent conversation.
	Continue bool `json:"continue,omitempty"`

	// ResumeSessionID resumes the exact conversation.
	ResumeSessionID string `json:"resumeSessionId,omitempty"`
}

// startCommandOptions configure session.startCommand.
type startCommandOptions struct {
	prompt         harness.PromptContent
	settings       harness.TurnSettings
	responseFormat *model.ResponseFormat
	resume         resumeFields
}

// startCommand builds a turn's start command.
func (s *session) startCommand(options startCommandOptions) startCommand {
	settings := s.harness.settings

	return startCommand{
		StartBase: bridgeadapter.NewStartBase(bridgeadapter.StartOptions{
			Model:                options.settings.Model,
			Tools:                options.settings.Tools,
			ResponseFormat:       options.responseFormat,
			PermissionMode:       s.permissionMode,
			BuiltinToolFiltering: s.builtinToolFiltering,
			Observability:        s.observability,
		}),
		Prompt:                 options.prompt,
		Instructions:           options.settings.Instructions,
		Thinking:               s.harness.thinking,
		Effort:                 settings.Effort,
		MaxTurns:               settings.MaxTurns,
		AgentProgressSummaries: settings.AgentProgressSummaries,
		ForwardSubagentText:    settings.ForwardSubagentText,
		Env:                    s.env,
		Skills:                 skillNames(options.settings.Skills),
		MCPServers:             settings.MCPServers,
		resumeFields:           options.resume,
	}
}

// skillNames returns the names of the skills.
func skillNames(skills []harness.Skill) []string {
	names := make([]string, 0, len(skills))
	for _, skill := range skills {
		names = append(names, skill.Name)
	}

	return names
}
