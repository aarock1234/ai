package bedrock

import (
	"net/http"
	"strings"
)

// StreamErrorStatus is the HTTP status and retry behavior of a Bedrock
// stream exception.
type StreamErrorStatus struct {
	StatusCode int
	Retryable  bool
}

// streamErrorStatuses are keyed by the event names of the exceptions.
var streamErrorStatuses = map[string]StreamErrorStatus{
	"internalServerException": {
		StatusCode: http.StatusInternalServerError,
		Retryable:  true,
	},
	"modelStreamErrorException": {
		StatusCode: http.StatusFailedDependency,
		Retryable:  true,
	},
	"serviceUnavailableException": {
		StatusCode: http.StatusServiceUnavailable,
		Retryable:  true,
	},
	"throttlingException": {
		StatusCode: http.StatusTooManyRequests,
		Retryable:  true,
	},
	"validationException": {
		StatusCode: http.StatusBadRequest,
		Retryable:  false,
	},
}

// StreamErrorStatusOf returns the status of a stream exception and reports
// whether the type is known. Both the event name, such as
// "throttlingException", and the exception name, such as
// "ThrottlingException", are accepted.
func StreamErrorStatusOf(errorType string) (StreamErrorStatus, bool) {
	if status, known := streamErrorStatuses[errorType]; known {
		return status, true
	}

	if errorType == "" {
		return StreamErrorStatus{}, false
	}

	eventName := strings.ToLower(errorType[:1]) + errorType[1:]
	if eventName == errorType {
		return StreamErrorStatus{}, false
	}

	status, known := streamErrorStatuses[eventName]

	return status, known
}
