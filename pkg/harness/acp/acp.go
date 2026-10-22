// Package acp ports the experimental upstream @ai-sdk/harness-acp package: a
// harness adapter backed by an Agent Client Protocol (version 1)
// implementation, such as Codex ACP, installed inside the sandbox.
//
// The adapter ships a bridge process that runs inside the sandbox next to
// the implementation and talks to the host over a WebSocket on a
// sandbox-proxied port; the bridge is the upstream JavaScript bundle,
// embedded verbatim and written by the bootstrap recipe. A bridge-backed ACP
// harness requires a sandbox with at least one exposed port, or explicit
// Settings.Port and Settings.PortEndpoint.
//
// Go adaptations: context.Context replaces abort signals; settings are one
// Settings struct; optional strings and slices use their zero values for
// unset; and identities sort object keys by byte order.
package acp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"slices"

	"github.com/aarock1234/ai/pkg/harness"
)

// Version is the upstream @ai-sdk/harness-acp version whose bridge this
// package embeds. It is the default client app version.
const Version = "1.0.96"

// defaultClientAppName is the default client app name.
const defaultClientAppName = "ai-sdk-harness-acp"

// hostToolsServerName is the MCP server name reserved for application tools.
const hostToolsServerName = "ai-sdk-harness-tools"

// harnessIDPattern accepts stable kebab-case harness identifiers.
var harnessIDPattern = regexp.MustCompile(`^[a-z0-9]+(?:-[a-z0-9]+)*$`)

var (
	_ harness.Harness                       = (*Harness)(nil)
	_ harness.Bootstrapper                  = (*Harness)(nil)
	_ harness.LifecycleStateSchemaProvider  = (*Harness)(nil)
	_ harness.BuiltinToolApprovalSupporter  = (*Harness)(nil)
	_ harness.BuiltinToolFilteringSupporter = (*Harness)(nil)
	_ harness.Session                       = (*session)(nil)
	_ harness.PromptControl                 = (*turn)(nil)
	_ harness.ToolApprovalSubmitter         = (*turn)(nil)
)

// Harness is an ACP harness adapter. It implements harness.Harness with
// bootstrap, lifecycle state validation and built-in tool approvals.
type Harness struct {
	settings       Settings
	clientApp      ClientApp
	builtinTools   []harness.BuiltinTool
	implementation implementation
	bootstrap      harness.Bootstrap
}

// New validates the settings and returns the adapter.
func New(settings Settings) (*Harness, error) {
	if err := validateSettings(settings); err != nil {
		return nil, err
	}

	source := newImplementation(settings)
	if err := source.validate(); err != nil {
		return nil, err
	}

	bootstrap, err := newBootstrap(settings.HarnessID, source)
	if err != nil {
		return nil, err
	}

	builtinTools, err := resolveBuiltinTools(settings)
	if err != nil {
		return nil, err
	}

	clientApp := settings.ClientApp
	if clientApp == (ClientApp{}) {
		clientApp = ClientApp{
			Name:    defaultClientAppName,
			Version: Version,
		}
	}

	return &Harness{
		settings:       settings,
		clientApp:      clientApp,
		builtinTools:   builtinTools,
		implementation: source,
		bootstrap:      bootstrap,
	}, nil
}

// HarnessID returns the configured harness ID.
func (h *Harness) HarnessID() string {
	return h.settings.HarnessID
}

// BuiltinTools returns the configured built-in tools, with the common
// askUserQuestions tool when AskUserQuestions is configured.
func (h *Harness) BuiltinTools() []harness.BuiltinTool {
	return slices.Clone(h.builtinTools)
}

// Bootstrap returns the recipe that installs the bridge and the
// implementation.
func (h *Harness) Bootstrap(context.Context) (*harness.Bootstrap, error) {
	bootstrap := h.bootstrap

	return &bootstrap, nil
}

// LifecycleStateSchema returns the JSON Schema of the lifecycle state data.
func (h *Harness) LifecycleStateSchema() json.RawMessage {
	return slices.Clone(lifecycleStateSchema)
}

// SupportsBuiltinToolApprovals reports true: the bridge turns ACP
// permission requests into approval requests.
func (h *Harness) SupportsBuiltinToolApprovals() bool {
	return true
}

// SupportsBuiltinToolFiltering reports false: ACP v1 cannot hide built-in
// tools.
func (h *Harness) SupportsBuiltinToolFiltering() bool {
	return false
}

// validateSettings checks the settings that do not describe the
// implementation.
func validateSettings(settings Settings) error {
	if settings.Version != "" && settings.Version != "v1" {
		return fmt.Errorf("unsupported acp protocol version %q; supported versions: \"v1\"", settings.Version)
	}

	if (settings.CredentialEnv == nil) != (settings.CredentialBrokering == nil) {
		return errors.New("acp credentialEnv and credentialBrokering must be configured together")
	}

	if _, reserved := settings.MCPServers[hostToolsServerName]; reserved {
		return fmt.Errorf("acp mcp server name %q is reserved for harness agent tools", hostToolsServerName)
	}

	if !harnessIDPattern.MatchString(settings.HarnessID) {
		return fmt.Errorf("acp harnessId must be a stable kebab-case identifier; received %q", settings.HarnessID)
	}

	validators := []func(settings Settings) error{
		validateModelMapping,
		validatePermissionModeMapping,
		validateInstructionMapping,
		validateOutputSchemaMapping,
		validateProviderAuthentication,
		validateAskUserQuestions,
	}

	for _, validate := range validators {
		if err := validate(settings); err != nil {
			return err
		}
	}

	return nil
}

// validateModelMapping checks the required model mapping.
func validateModelMapping(settings Settings) error {
	mapping := settings.ModelMapping

	if mapping.Type != ModelSessionConfigOption && mapping.Type != ModelSessionModel {
		return fmt.Errorf("acp modelMapping type %q is not supported", mapping.Type)
	}

	if mapping.Path == "" {
		return errors.New("acp modelMapping path must not be empty")
	}

	return nil
}

// validatePermissionModeMapping checks each permission mode target.
func validatePermissionModeMapping(settings Settings) error {
	mapping := settings.PermissionModeMapping
	if mapping == nil {
		return nil
	}

	targets := []*PermissionModeTarget{
		mapping.AllowReads,
		mapping.AllowEdits,
		mapping.AllowAll,
	}

	for _, target := range targets {
		if target != nil && !isValidPermissionModeTarget(*target) {
			return fmt.Errorf("acp permission mode target %+v is invalid", *target)
		}
	}

	return nil
}

// isValidPermissionModeTarget reports a session mode with a mode ID, or a
// configuration option with a string or bool value.
func isValidPermissionModeTarget(target PermissionModeTarget) bool {
	switch target.Type {
	case TargetSessionMode:
		return target.ModeID != ""
	case TargetSessionConfigOption:
		switch target.Value.(type) {
		case string, bool:
			return target.ConfigID != ""
		default:
			return false
		}
	default:
		return false
	}
}

// validateInstructionMapping checks the instruction mapping's fields.
func validateInstructionMapping(settings Settings) error {
	mapping := settings.InstructionMapping
	if mapping == nil {
		return nil
	}

	switch mapping.Type {
	case InstructionsFilesystem:
		if mapping.File == "" {
			return errors.New("acp filesystem instruction mapping requires a file")
		}

		return nil
	case InstructionsLaunchEnvJSON:
		if mapping.Variable == "" {
			return errors.New("acp launch-env-json instruction mapping requires a variable")
		}

		return validateMappingPath("instruction mapping", mapping.Path)
	case InstructionsSessionMeta:
		return validateMappingPath("instruction mapping", mapping.Path)
	default:
		return fmt.Errorf("acp instruction mapping type %q is not supported", mapping.Type)
	}
}

// validateOutputSchemaMapping checks the output schema mapping.
func validateOutputSchemaMapping(settings Settings) error {
	mapping := settings.OutputSchemaMapping
	if mapping == nil {
		return nil
	}

	if mapping.Type != OutputSchemaPromptMeta {
		return fmt.Errorf("acp output schema mapping type %q is not supported", mapping.Type)
	}

	return validateMappingPath("output schema mapping", mapping.Path)
}

// validateMappingPath requires a non-empty path without empty segments.
func validateMappingPath(name string, mappingPath []string) error {
	if len(mappingPath) == 0 || slices.Contains(mappingPath, "") {
		return fmt.Errorf("acp %s path must contain non-empty segments", name)
	}

	return nil
}

// validateProviderAuthentication checks that the gateway profile values
// encode to JSON.
func validateProviderAuthentication(settings Settings) error {
	if settings.ProviderAuthentication == nil {
		return nil
	}

	return validateProfileValue(settings.ProviderAuthentication.Gateway.Env)
}

// validateAskUserQuestions checks the question request mapping.
func validateAskUserQuestions(settings Settings) error {
	questions := settings.AskUserQuestions
	if questions == nil {
		return nil
	}

	if questions.RequestMethod == "" {
		return errors.New("acp askUserQuestions requires a request method")
	}

	if questions.FromNativeRequest == nil || questions.ToNativeResponse == nil {
		return errors.New("acp askUserQuestions requires FromNativeRequest and ToNativeResponse")
	}

	return nil
}

// resolveBuiltinTools returns the configured built-in tools, adding the
// common askUserQuestions tool when question requests are mapped.
func resolveBuiltinTools(settings Settings) ([]harness.BuiltinTool, error) {
	tools := slices.Clone(settings.BuiltinTools)
	if settings.AskUserQuestions == nil {
		return tools, nil
	}

	for _, common := range harness.CommonTools() {
		if common.Name != string(harness.ToolAskUserQuestions) {
			continue
		}

		tools = slices.DeleteFunc(tools, func(tool harness.BuiltinTool) bool {
			return tool.Name == common.Name
		})

		return append(tools, harness.BuiltinTool{
			Tool:       common,
			CommonName: harness.ToolAskUserQuestions,
		}), nil
	}

	return nil, errors.New("acp askUserQuestions: the common askUserQuestions tool is unavailable")
}
