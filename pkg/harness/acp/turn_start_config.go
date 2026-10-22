package acp

import (
	"slices"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/bridge"
)

// turnStartConfigVersion is the version of persisted turn start
// configurations.
const turnStartConfigVersion = 1

// turnStartConfig is the non-secret start configuration of a turn. It is
// persisted with lifecycle state, so that a replacement bridge can rerun or
// restore the turn, and its fingerprint must match the current settings.
type turnStartConfig struct {
	Version                  int                        `json:"version"`
	ConfigurationFingerprint string                     `json:"configurationFingerprint"`
	Prompt                   []textBlock                `json:"prompt"`
	Tools                    []harness.ToolSpec         `json:"tools"`
	BuiltinTools             []builtinToolMapping       `json:"builtinTools"`
	PermissionMode           harness.PermissionMode     `json:"permissionMode"`
	PermissionModeMapping    *PermissionModeMapping     `json:"permissionModeMapping,omitempty"`
	Model                    string                     `json:"model,omitempty"`
	ModelMapping             *ModelMapping              `json:"modelMapping,omitempty"`
	Debug                    *harness.DebugConfig       `json:"debug,omitempty"`
	ResponseFormat           *bridge.ResponseFormatWire `json:"responseFormat,omitempty"`
	OutputSchemaMapping      *OutputSchemaMapping       `json:"outputSchemaMapping,omitempty"`
}

// coldSessionState is the part of the turn start configuration a cold
// session restoration needs: the turn start configuration without the
// prompt, debug and model settings.
type coldSessionState struct {
	Version                  int                        `json:"version"`
	ConfigurationFingerprint string                     `json:"configurationFingerprint"`
	Tools                    []harness.ToolSpec         `json:"tools"`
	BuiltinTools             []builtinToolMapping       `json:"builtinTools"`
	PermissionMode           harness.PermissionMode     `json:"permissionMode"`
	PermissionModeMapping    *PermissionModeMapping     `json:"permissionModeMapping,omitempty"`
	ResponseFormat           *bridge.ResponseFormatWire `json:"responseFormat,omitempty"`
	OutputSchemaMapping      *OutputSchemaMapping       `json:"outputSchemaMapping,omitempty"`
}

// sessionConfiguration is the non-secret session configuration every turn
// start configuration is fingerprinted with.
type sessionConfiguration struct {
	authenticationProfile authenticationProfile
	sessionMeta           map[string]any
	instructionMapping    *InstructionMapping
	outputSchemaMapping   *OutputSchemaMapping
	modelMapping          ModelMapping
	builtinTools          []builtinToolMapping
	permissionModeMapping *PermissionModeMapping
	mcpServers            map[string]any
}

// turnStartInput is the per-turn part of a turn start configuration.
type turnStartInput struct {
	prompt         []textBlock
	tools          []harness.ToolSpec
	permissionMode harness.PermissionMode
	debug          *harness.DebugConfig
	responseFormat *bridge.ResponseFormatWire
	model          string
}

// newTurnStartConfig returns the turn start configuration of a turn.
func (c sessionConfiguration) newTurnStartConfig(input turnStartInput) (turnStartConfig, error) {
	fingerprint, err := c.fingerprint()
	if err != nil {
		return turnStartConfig{}, err
	}

	tools, err := serializableToolSpecs(input.tools)
	if err != nil {
		return turnStartConfig{}, err
	}

	prompt := slices.Clone(input.prompt)
	if prompt == nil {
		prompt = []textBlock{}
	}

	config := turnStartConfig{
		Version:                  turnStartConfigVersion,
		ConfigurationFingerprint: fingerprint,
		Prompt:                   prompt,
		Tools:                    tools,
		BuiltinTools:             slices.Clone(c.builtinTools),
		PermissionMode:           input.permissionMode,
		PermissionModeMapping:    c.permissionModeMapping,
		Debug:                    input.debug,
		ResponseFormat:           input.responseFormat,
	}

	if input.model != "" {
		modelMapping := c.modelMapping
		config.Model = input.model
		config.ModelMapping = &modelMapping
	}

	if c.outputSchemaMapping != nil {
		config.OutputSchemaMapping = &OutputSchemaMapping{
			Type: c.outputSchemaMapping.Type,
			Path: slices.Clone(c.outputSchemaMapping.Path),
		}
	}

	return config, nil
}

// fingerprint digests the session configuration.
func (c sessionConfiguration) fingerprint() (string, error) {
	payload := map[string]any{
		"authenticationProfile": c.authenticationProfile,
		"sessionMeta":           nullable(c.sessionMeta),
		"modelMapping":          c.modelMapping,
		"builtinTools":          c.builtinTools,
		"permissionModeMapping": nullablePointer(c.permissionModeMapping),
		"mcpServers":            nullable(c.mcpServers),
	}

	if c.instructionMapping != nil {
		payload["instructionMapping"] = c.instructionMapping
	}

	if c.outputSchemaMapping != nil {
		payload["outputSchemaMapping"] = c.outputSchemaMapping
	}

	return stableFingerprint(payload)
}

// coldSession returns the cold session state of a turn start
// configuration.
func (c turnStartConfig) coldSession() coldSessionState {
	return coldSessionState{
		Version:                  c.Version,
		ConfigurationFingerprint: c.ConfigurationFingerprint,
		Tools:                    c.Tools,
		BuiltinTools:             c.BuiltinTools,
		PermissionMode:           c.PermissionMode,
		PermissionModeMapping:    c.PermissionModeMapping,
		ResponseFormat:           c.ResponseFormat,
		OutputSchemaMapping:      c.OutputSchemaMapping,
	}
}
