package codex

import (
	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/bridge"
	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
	"github.com/aarock1234/ai/pkg/model"
)

// startCommand starts a turn in the Codex bridge. Its Prompt replaces the
// text prompt of the shared start fields.
type startCommand struct {
	bridge.StartBase

	Prompt       harness.PromptContent `json:"prompt"`
	Instructions string                `json:"instructions,omitempty"`

	ReasoningEffort ReasoningEffort   `json:"reasoningEffort,omitempty"`
	WebSearch       *bool             `json:"webSearch,omitempty"`
	CodexConfig     map[string]any    `json:"codexConfig,omitempty"`
	MCPServers      map[string]any    `json:"mcpServers,omitempty"`
	Headers         map[string]string `json:"headers,omitempty"`

	// ResumeThreadID makes the bridge resume the app-server thread with
	// thread/resume instead of starting a fresh thread.
	ResumeThreadID string `json:"resumeThreadId,omitempty"`

	// RestartThread makes the bridge start a fresh thread, because the
	// thread's configuration changed.
	RestartThread bool `json:"restartThread,omitempty"`
}

// startCommandOptions configure session.startCommand.
type startCommandOptions struct {
	prompt         harness.PromptContent
	settings       harness.TurnSettings
	responseFormat *model.ResponseFormat
	resumeThreadID string
	restartThread  bool
}

// startCommand builds a turn's start command.
func (s *session) startCommand(options startCommandOptions) startCommand {
	settings := s.harness.settings

	turnModel := options.settings.Model
	if turnModel == "" {
		turnModel = DefaultModel
	}

	return startCommand{
		StartBase: bridgeadapter.NewStartBase(bridgeadapter.StartOptions{
			Model:                turnModel,
			Tools:                options.settings.Tools,
			ResponseFormat:       options.responseFormat,
			PermissionMode:       s.permissionMode,
			BuiltinToolFiltering: s.builtinToolFiltering,
			Observability:        s.observability,
		}),
		Prompt:          options.prompt,
		Instructions:    options.settings.Instructions,
		ReasoningEffort: settings.ReasoningEffort,
		WebSearch:       settings.WebSearch,
		CodexConfig:     settings.CodexConfig,
		MCPServers:      settings.MCPServers,
		Headers:         s.headers,
		ResumeThreadID:  options.resumeThreadID,
		RestartThread:   options.restartThread,
	}
}
