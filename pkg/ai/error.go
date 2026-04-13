package ai

import (
	"errors"

	"github.com/aarock1234/ai/internal/apierror"
)

// Sentinel errors for expected conditions.
var (
	// ErrNoContent is returned when the model produces an empty response.
	ErrNoContent = errors.New("no content in response")

	// ErrMaxStepsExceeded is returned when the tool loop reaches the
	// step limit without a final response.
	ErrMaxStepsExceeded = errors.New("max steps exceeded")

	// ErrInvalidMaxSteps is returned when WithMaxSteps is less than 1.
	ErrInvalidMaxSteps = errors.New("max steps must be at least 1")

	// ErrSchemaValidation is returned when structured output fails to
	// match the expected schema.
	ErrSchemaValidation = errors.New("schema validation failed")
)

// APIError is an alias for apierror.APIError, kept here for backward
// compatibility so consumers can reference ai.APIError.
type APIError = apierror.APIError

// NewAPIError creates an APIError. See apierror.New.
var NewAPIError = apierror.New
