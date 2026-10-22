package harness

import (
	"fmt"

	"github.com/aarock1234/ai/internal/schema"
)

// lifecycleStateHeader is the common part of both lifecycle states.
type lifecycleStateHeader struct {
	stateType            LifecycleStateType
	harnessID            string
	specificationVersion string
}

// validateResumeSessionState checks a resume state against the harness and
// validates its data and nested continuation.
func validateResumeSessionState(harness Harness, state ResumeSessionState) (ResumeSessionState, error) {
	err := validateLifecycleHeader(harness, StateResumeSession, lifecycleStateHeader{
		stateType:            state.Type,
		harnessID:            state.HarnessID,
		specificationVersion: state.SpecificationVersion,
	})
	if err != nil {
		return ResumeSessionState{}, err
	}

	if err := validateLifecycleData(harness, state.Data); err != nil {
		return ResumeSessionState{}, err
	}

	if state.ContinueFrom == nil {
		return state, nil
	}

	continueFrom, err := validateContinueTurnState(harness, *state.ContinueFrom)
	if err != nil {
		return ResumeSessionState{}, err
	}

	state.ContinueFrom = &continueFrom

	return state, nil
}

// validateContinueTurnState checks a continuation state against the harness
// and validates its data.
func validateContinueTurnState(harness Harness, state ContinueTurnState) (ContinueTurnState, error) {
	err := validateLifecycleHeader(harness, StateContinueTurn, lifecycleStateHeader{
		stateType:            state.Type,
		harnessID:            state.HarnessID,
		specificationVersion: state.SpecificationVersion,
	})
	if err != nil {
		return ContinueTurnState{}, err
	}

	if err := validateLifecycleData(harness, state.Data); err != nil {
		return ContinueTurnState{}, err
	}

	return state, nil
}

// validateLifecycleHeader checks the type, version and harness of a state.
func validateLifecycleHeader(harness Harness, expectedType LifecycleStateType, header lifecycleStateHeader) error {
	if header.stateType != expectedType {
		return &Error{
			Message: fmt.Sprintf("Lifecycle state has unexpected type '%s'; expected '%s'.", header.stateType, expectedType),
		}
	}

	if header.specificationVersion != SpecificationVersion {
		return &Error{
			Message: fmt.Sprintf(
				"Lifecycle state has unexpected specificationVersion '%s'; expected '%s'.",
				header.specificationVersion,
				SpecificationVersion,
			),
		}
	}

	if header.harnessID != harness.HarnessID() {
		return &Error{
			Message: fmt.Sprintf(
				"Lifecycle state was produced by harness '%s' but this agent uses '%s'.",
				header.harnessID,
				harness.HarnessID(),
			),
		}
	}

	return nil
}

// validateLifecycleData validates state data against the harness's schema,
// when it declares one.
func validateLifecycleData(harness Harness, data []byte) error {
	provider, ok := harness.(LifecycleStateSchemaProvider)
	if !ok {
		return nil
	}

	validator, err := schema.Compile(provider.LifecycleStateSchema())
	if err != nil {
		return fmt.Errorf("compile lifecycle state schema of harness %q: %w", harness.HarnessID(), err)
	}

	if len(data) == 0 {
		data = []byte("null")
	}

	if err := validator.Validate(data); err != nil {
		return &Error{
			Message: fmt.Sprintf("Lifecycle state failed schema validation for harness '%s': %v", harness.HarnessID(), err),
			Cause:   err,
		}
	}

	return nil
}
