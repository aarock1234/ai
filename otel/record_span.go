package otel

import (
	"errors"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// recordError records err on span as an exception event, sets the error
// status, and records the HTTP status code of a failed provider call. For a
// retry error the status code comes from the last attempt.
func recordError(span trace.Span, err error) {
	if statusCode, ok := apiCallStatusCode(err); ok {
		span.SetAttributes(attribute.Int("http.response.status_code", statusCode))
	}

	span.RecordError(err)
	span.SetStatus(codes.Error, err.Error())
}

// recordLegacyError records err on span without the HTTP status code, as
// the legacy integration does.
func recordLegacyError(span trace.Span, err error) {
	span.RecordError(err)
	span.SetStatus(codes.Error, err.Error())
}

func apiCallStatusCode(err error) (int, bool) {
	if retryErr, ok := errors.AsType[*ai.RetryError](err); ok {
		err = retryErr.LastError()
	}

	var apiCallErr *model.APICallError
	if !errors.As(err, &apiCallErr) || apiCallErr.StatusCode == 0 {
		return 0, false
	}

	return apiCallErr.StatusCode, true
}

// markErrorFinish sets the error status on span when the model finished
// with an error.
func markErrorFinish(span trace.Span, finishReason model.UnifiedFinishReason) {
	if finishReason == model.FinishError {
		span.SetStatus(codes.Error, "")
	}
}
