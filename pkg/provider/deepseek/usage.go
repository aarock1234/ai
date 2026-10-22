package deepseek

import (
	"bytes"
	"encoding/json"
	"net/http"
	"strconv"
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

// ChatMetadata is the result provider metadata of chat models, stored under
// the "deepseek" namespace.
type ChatMetadata struct {
	// PromptCacheHitTokens and PromptCacheMissTokens split the prompt tokens
	// by context cache result.
	PromptCacheHitTokens  *int `json:"promptCacheHitTokens,omitempty"`
	PromptCacheMissTokens *int `json:"promptCacheMissTokens,omitempty"`

	// ResponseObject is the response object type, such as
	// "chat.completion".
	ResponseObject string `json:"responseObject,omitempty"`

	// ChoiceIndex is the index of the returned choice.
	ChoiceIndex *int `json:"choiceIndex,omitempty"`

	// MessageRole is the role of the returned message.
	MessageRole string `json:"messageRole,omitempty"`

	// ToolCallTypes are the types of the returned tool calls, in order.
	ToolCallTypes []string `json:"toolCallTypes,omitempty"`

	// Logprobs are the token log probabilities, when requested.
	Logprobs *Logprobs `json:"logprobs,omitempty"`

	// SystemFingerprint identifies the backend configuration.
	SystemFingerprint string `json:"systemFingerprint,omitempty"`
}

// convertUsage converts DeepSeek usage. Cache hits are reported as cache
// reads.
func convertUsage(usage *chatUsage) model.Usage {
	if usage == nil {
		return model.Usage{}
	}

	promptTokens := valueOrZero(usage.PromptTokens)
	completionTokens := valueOrZero(usage.CompletionTokens)

	cacheReadTokens := 0
	switch {
	case usage.PromptCacheHitTokens != nil:
		cacheReadTokens = *usage.PromptCacheHitTokens
	case usage.PromptTokensDetails != nil && usage.PromptTokensDetails.CachedTokens != nil:
		cacheReadTokens = *usage.PromptTokensDetails.CachedTokens
	}

	reasoningTokens := 0
	if usage.CompletionDetails != nil {
		reasoningTokens = valueOrZero(usage.CompletionDetails.ReasoningTokens)
	}

	return model.Usage{
		InputTokens: model.InputTokens{
			Total:     new(promptTokens),
			NoCache:   new(promptTokens - cacheReadTokens),
			CacheRead: new(cacheReadTokens),
		},
		OutputTokens: model.OutputTokens{
			Total:     new(completionTokens),
			Text:      new(max(0, completionTokens-reasoningTokens)),
			Reasoning: new(reasoningTokens),
		},
		Raw: usage.Raw,
	}
}

// cacheTokens returns the cache hit and miss token counts of the usage.
func cacheTokens(usage *chatUsage) (*int, *int) {
	if usage == nil {
		return nil, nil
	}

	return usage.PromptCacheHitTokens, usage.PromptCacheMissTokens
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
	case "tool_calls":
		return model.FinishToolCalls
	case "insufficient_system_resource":
		return model.FinishError
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
// streamed error.
type streamErrorStatus struct {
	statusCode int
	retryable  bool
}

// streamErrorStatuses maps streamed error codes and types to the status
// that the same failure would have produced before the stream started.
var streamErrorStatuses = map[string]streamErrorStatus{
	"rate_limit_exceeded": {
		statusCode: http.StatusTooManyRequests,
		retryable:  true,
	},
	"rate_limit_error": {
		statusCode: http.StatusTooManyRequests,
		retryable:  true,
	},
	"server_error": {
		statusCode: http.StatusInternalServerError,
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

// insufficientQuota is the code and type of an exhausted balance, which
// retrying does not fix.
const insufficientQuota = "insufficient_quota"

// streamOrigin identifies the HTTP exchange of a stream, for errors.
type streamOrigin struct {
	url     string
	headers http.Header
}

// apiError converts a streamed error event into an API call error, so that
// retry decisions treat it like an HTTP failure.
func (o streamOrigin) apiError(detail *ErrorDetail, data string) *model.APICallError {
	status := streamStatus(errorCode(detail.Code), valueOrEmpty(detail.Type))

	return &model.APICallError{
		Message:         detail.Message,
		URL:             o.url,
		StatusCode:      status.statusCode,
		ResponseHeaders: o.headers,
		ResponseBody:    data,
		IsRetryable:     status.retryable,
	}
}

// streamStatus infers the status of a streamed error: an exhausted quota, an
// explicit HTTP status code, then the code and the type.
func streamStatus(code string, errorType string) streamErrorStatus {
	if code == insufficientQuota || errorType == insufficientQuota {
		return streamErrorStatus{
			statusCode: http.StatusTooManyRequests,
		}
	}

	if statusCode, ok := httpStatusCode(code); ok {
		return streamErrorStatus{
			statusCode: statusCode,
			retryable:  model.RetryableStatus(statusCode),
		}
	}

	if status, ok := streamErrorStatuses[code]; ok {
		return status
	}

	return streamErrorStatuses[errorType]
}

// httpStatusCode reads a code that is an HTTP error status.
func httpStatusCode(code string) (int, bool) {
	if len(code) != 3 {
		return 0, false
	}

	statusCode, err := strconv.Atoi(code)
	if err != nil || statusCode < 400 || statusCode > 599 {
		return 0, false
	}

	return statusCode, true
}

// errorCode returns an error code, a string or a number, as text.
func errorCode(raw json.RawMessage) string {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return ""
	}

	var text string
	if err := json.Unmarshal(trimmed, &text); err == nil {
		return text
	}

	return string(trimmed)
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
