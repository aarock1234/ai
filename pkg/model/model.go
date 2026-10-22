// Package model defines the provider contracts shared by every adapter and by
// the consumer-facing ai package. The shapes mirror the Vercel AI SDK
// provider specification (version 4) using Go types: tagged unions become
// sealed interfaces, optional numbers become pointers, and provider-specific
// state travels in namespaced JSON objects.
//
// Provider packages implement these interfaces. Applications normally use the
// ai package instead of calling models directly.
package model

import (
	"context"
	"regexp"
)

// LanguageModel generates text, reasoning, files and tool calls from a prompt.
type LanguageModel interface {
	// Provider returns the provider identifier, such as "openai.responses".
	Provider() string

	// ModelID returns the provider-specific model identifier.
	ModelID() string

	// SupportedURLs maps media type patterns (for example "image/*") to the
	// URL patterns the provider downloads itself. File URLs that do not match
	// are downloaded by the caller and sent inline.
	SupportedURLs() map[string][]*regexp.Regexp

	// Generate performs one non-streaming generation call.
	Generate(ctx context.Context, options CallOptions) (*GenerateResult, error)

	// Stream performs one streaming generation call. The caller must either
	// consume the returned parts or call Close.
	Stream(ctx context.Context, options CallOptions) (*StreamResult, error)
}
