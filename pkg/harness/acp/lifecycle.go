package acp

import (
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/aarock1234/ai/pkg/harness"
)

// lifecycleStateSchema is the JSON Schema of lifecycleData.
//
//go:embed lifecycle_state_schema.json
var lifecycleStateSchema []byte

// lifecycleData is the adapter data of ACP lifecycle states.
type lifecycleData struct {
	ImplementationIdentity string                 `json:"implementationIdentity"`
	AuthenticationProfile  *authenticationProfile `json:"authenticationProfile,omitempty"`

	// SandboxCredentialEnvironment holds the credential placeholders, which
	// a resumed session reuses so that installed request transformations
	// keep matching.
	SandboxCredentialEnvironment map[string]string `json:"sandboxCredentialEnvironment,omitempty"`

	ACPSessionID    string             `json:"acpSessionId,omitempty"`
	Bridge          *bridgeCoordinates `json:"bridge,omitempty"`
	ColdSession     *coldSessionState  `json:"coldSession,omitempty"`
	TurnStartConfig *turnStartConfig   `json:"turnStartConfig,omitempty"`
	Recovery        *recoveryStatus    `json:"recovery,omitempty"`
	Restoration     *restoration       `json:"restoration,omitempty"`

	InitialGuidanceApplied  *bool  `json:"initialGuidanceApplied,omitempty"`
	InstructionsFingerprint string `json:"instructionsFingerprint,omitempty"`
	SkillsDirectory         string `json:"skillsDirectory,omitempty"`
}

// bridgeCoordinates locate a running bridge.
type bridgeCoordinates struct {
	Port            int    `json:"port"`
	Token           string `json:"token"`
	LastSeenEventID int64  `json:"lastSeenEventId"`
	SandboxID       string `json:"sandboxId,omitempty"`
	StateDir        string `json:"stateDir,omitempty"`
}

// recoveryStatus records how a turn was recovered after the bridge process
// was lost.
type recoveryStatus struct {
	// Mode is recoveryDiskReplay or recoveryLossyRerun.
	Mode   string `json:"mode"`
	Reason string `json:"reason"`
}

// Cold restoration methods.
const (
	restoreByResume = "resume"
	restoreByLoad   = "load"
)

// restoration records how a cold session was restored.
type restoration struct {
	Method string `json:"method"`
}

// decodeLifecycleData decodes lifecycle state data, applying the upstream
// normalizations.
func decodeLifecycleData(data json.RawMessage) (lifecycleData, error) {
	if len(data) == 0 {
		return lifecycleData{}, errors.New("acp lifecycle state data is missing")
	}

	var decoded lifecycleData
	if err := json.Unmarshal(data, &decoded); err != nil {
		return lifecycleData{}, fmt.Errorf("decode acp lifecycle state data: %w", err)
	}

	if decoded.AuthenticationProfile != nil && decoded.AuthenticationProfile.ProviderKind == "implementation-default" {
		decoded.AuthenticationProfile.ProviderKind = providerDirect
	}

	if decoded.TurnStartConfig != nil {
		decoded.TurnStartConfig.Tools = nonNil(decoded.TurnStartConfig.Tools)
		decoded.TurnStartConfig.BuiltinTools = nonNil(decoded.TurnStartConfig.BuiltinTools)
		decoded.TurnStartConfig.Prompt = nonNil(decoded.TurnStartConfig.Prompt)
	}

	if decoded.ColdSession != nil {
		decoded.ColdSession.Tools = nonNil(decoded.ColdSession.Tools)
		decoded.ColdSession.BuiltinTools = nonNil(decoded.ColdSession.BuiltinTools)
	}

	return decoded, nil
}

// encode encodes lifecycle state data.
func (d lifecycleData) encode() (json.RawMessage, error) {
	encoded, err := marshalJSON(d)
	if err != nil {
		return nil, fmt.Errorf("encode acp lifecycle state data: %w", err)
	}

	return encoded, nil
}

// initialGuidanceApplied reports whether the instruction guidance of the
// first prompt was already applied: never for a fresh session, and by
// default for resumed state that predates the field.
func (d *lifecycleData) initialGuidanceApplied() bool {
	if d == nil {
		return false
	}

	if d.InitialGuidanceApplied == nil {
		return true
	}

	return *d.InitialGuidanceApplied
}

// compatibilityCheck is what lifecycle state must match.
type compatibilityCheck struct {
	harnessID              string
	stateHarnessID         string
	implementationIdentity string
	authenticationProfile  authenticationProfile
	sandboxID              string
}

// validateCompatibility rejects lifecycle state produced by another
// harness, implementation, authentication profile or sandbox.
func (d lifecycleData) validateCompatibility(check compatibilityCheck) error {
	if check.stateHarnessID != check.harnessID {
		return fmt.Errorf("acp lifecycle state was produced by harness %q, but this harness is %q", check.stateHarnessID, check.harnessID)
	}

	if d.ImplementationIdentity != check.implementationIdentity {
		return errors.New("acp lifecycle state is incompatible with the configured implementation")
	}

	if d.AuthenticationProfile == nil || d.AuthenticationProfile.Digest != check.authenticationProfile.Digest {
		return errors.New("acp lifecycle state is incompatible with the configured authentication profile")
	}

	if d.Bridge == nil || d.Bridge.SandboxID == "" || check.sandboxID == "" {
		return nil
	}

	if d.Bridge.SandboxID != check.sandboxID {
		return fmt.Errorf("acp lifecycle state belongs to sandbox %q, not %q", d.Bridge.SandboxID, check.sandboxID)
	}

	return nil
}

// newContinueTurnState returns a continuation state with the session data.
func newContinueTurnState(harnessID string, data lifecycleData) (harness.ContinueTurnState, error) {
	encoded, err := data.encode()
	if err != nil {
		return harness.ContinueTurnState{}, err
	}

	return harness.NewContinueTurnState(harnessID, encoded), nil
}

// newResumeSessionState returns a resume state with the session data.
func newResumeSessionState(harnessID string, data lifecycleData) (harness.ResumeSessionState, error) {
	encoded, err := data.encode()
	if err != nil {
		return harness.ResumeSessionState{}, err
	}

	return harness.NewResumeSessionState(harnessID, encoded), nil
}

// nonNil returns an empty slice for nil.
func nonNil[T any](values []T) []T {
	if values == nil {
		return []T{}
	}

	return values
}
