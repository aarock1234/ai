package model

import (
	"encoding/json"
	"net/http"
	"time"
)

// GenerateResult is the outcome of a non-streaming language model call.
type GenerateResult struct {
	// Content is the generated output in provider order.
	Content []Content

	FinishReason     FinishReason
	Usage            Usage
	ProviderMetadata ProviderMetadata
	Warnings         []Warning

	Request  RequestInfo
	Response ResponseInfo
}

// RequestInfo describes the HTTP request sent to the provider.
type RequestInfo struct {
	// Body is the JSON request body, when the provider records it.
	Body json.RawMessage
}

// ResponseMetadata identifies a provider response. Zero values mean the
// provider did not report the field.
type ResponseMetadata struct {
	ID        string
	Timestamp time.Time
	ModelID   string
}

// ResponseInfo describes the HTTP response received from the provider.
type ResponseInfo struct {
	ResponseMetadata

	Headers http.Header

	// Body is the JSON response body, when the provider records it.
	Body json.RawMessage
}
