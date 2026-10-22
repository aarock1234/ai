package groq

import (
	"net/http"
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

// convertUsage converts Groq usage. Cached and reasoning token counts are
// reported only when Groq sends them.
func convertUsage(usage *chatUsage) model.Usage {
	if usage == nil {
		return model.Usage{}
	}

	promptTokens := valueOrZero(usage.PromptTokens)
	completionTokens := valueOrZero(usage.CompletionTokens)

	converted := model.Usage{
		InputTokens: model.InputTokens{
			Total:   new(promptTokens),
			NoCache: new(promptTokens),
		},
		OutputTokens: model.OutputTokens{
			Total: new(completionTokens),
			Text:  new(completionTokens),
		},
		Raw: usage.Raw,
	}

	if usage.PromptTokensDetails != nil && usage.PromptTokensDetails.CachedTokens != nil {
		cacheReadTokens := *usage.PromptTokensDetails.CachedTokens
		converted.InputTokens.NoCache = new(promptTokens - cacheReadTokens)
		converted.InputTokens.CacheRead = new(cacheReadTokens)
	}

	if usage.CompletionTokensDetails != nil && usage.CompletionTokensDetails.ReasoningTokens != nil {
		reasoningTokens := *usage.CompletionTokensDetails.ReasoningTokens
		converted.OutputTokens.Text = new(max(0, completionTokens-reasoningTokens))
		converted.OutputTokens.Reasoning = new(reasoningTokens)
	}

	return converted
}

// mapFinishReason converts a chat completion finish reason.
func mapFinishReason(finishReason *string) model.FinishReason {
	raw := valueOrEmpty(finishReason)

	return model.FinishReason{
		Unified: unifiedFinishReason(raw),
		Raw:     raw,
	}
}

func unifiedFinishReason(finishReason string) model.UnifiedFinishReason {
	switch finishReason {
	case "stop":
		return model.FinishStop
	case "length":
		return model.FinishLength
	case "content_filter":
		return model.FinishContentFilter
	case "function_call", "tool_calls":
		return model.FinishToolCalls
	default:
		return model.FinishOther
	}
}

// responseMetadata converts the response identification fields.
func responseMetadata(id *string, modelID *string, created *float64) model.ResponseMetadata {
	metadata := model.ResponseMetadata{
		ID:      valueOrEmpty(id),
		ModelID: valueOrEmpty(modelID),
	}

	if created != nil {
		metadata.Timestamp = time.Unix(int64(*created), 0)
	}

	return metadata
}

// streamErrorStatus is the HTTP status and retry decision implied by a
// streamed error type.
type streamErrorStatus struct {
	statusCode int
	retryable  bool
}

// streamErrorStatuses maps streamed error types to the status that the same
// failure would have produced before the stream started.
var streamErrorStatuses = map[string]streamErrorStatus{
	"rate_limit_error": {
		statusCode: http.StatusTooManyRequests,
		retryable:  true,
	},
	"api_error": {
		statusCode: http.StatusInternalServerError,
		retryable:  true,
	},
	"internal_server_error": {
		statusCode: http.StatusInternalServerError,
		retryable:  true,
	},
	"server_error": {
		statusCode: http.StatusInternalServerError,
		retryable:  true,
	},
	"overloaded_error": {
		statusCode: http.StatusServiceUnavailable,
		retryable:  true,
	},
	"service_unavailable": {
		statusCode: http.StatusServiceUnavailable,
		retryable:  true,
	},
	"timeout": {
		statusCode: http.StatusGatewayTimeout,
		retryable:  true,
	},
	"timeout_error": {
		statusCode: http.StatusGatewayTimeout,
		retryable:  true,
	},
	"authentication_error": {
		statusCode: http.StatusUnauthorized,
	},
	"invalid_api_key": {
		statusCode: http.StatusUnauthorized,
	},
	"permission_error": {
		statusCode: http.StatusForbidden,
	},
	"not_found_error": {
		statusCode: http.StatusNotFound,
	},
	"model_not_found": {
		statusCode: http.StatusNotFound,
	},
	"bad_request": {
		statusCode: http.StatusBadRequest,
	},
	"context_length_exceeded": {
		statusCode: http.StatusBadRequest,
	},
	"invalid_request_error": {
		statusCode: http.StatusBadRequest,
	},
}

// streamOrigin identifies the HTTP exchange of a stream, for errors.
type streamOrigin struct {
	url     string
	headers http.Header
}

// apiError converts a streamed error event into an API call error, so that
// retry decisions treat it like an HTTP failure. Unknown error types have no
// status.
func (o streamOrigin) apiError(detail *ErrorDetail, data string) *model.APICallError {
	status := streamErrorStatuses[detail.Type]

	return &model.APICallError{
		Message:         detail.Message,
		URL:             o.url,
		StatusCode:      status.statusCode,
		ResponseHeaders: o.headers,
		ResponseBody:    data,
		IsRetryable:     status.retryable,
	}
}

func valueOrZero(value *int) int {
	if value == nil {
		return 0
	}

	return *value
}

func valueOrEmpty(value *string) string {
	if value == nil {
		return ""
	}

	return *value
}
