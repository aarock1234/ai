package azure

import "github.com/aarock1234/ai/pkg/provider/openai"

// Responses API options and metadata use the OpenAI shapes under the "azure"
// namespace. Responses models fall back to the "openai" options namespace
// when the "azure" namespace is absent.
type (
	// ResponsesOptions are the Responses API options of a call.
	ResponsesOptions = openai.ResponsesOptions

	// ResponsesSystemMessageOptions are the options of a system message.
	ResponsesSystemMessageOptions = openai.ResponsesSystemMessageOptions

	// ResponsesMetadata is the metadata of a Responses API result.
	ResponsesMetadata = openai.ResponsesMetadata

	// ResponsesReasoningMetadata is the metadata of a reasoning part.
	ResponsesReasoningMetadata = openai.ResponsesReasoningMetadata

	// ResponsesTextMetadata is the metadata of a text part.
	ResponsesTextMetadata = openai.ResponsesTextMetadata

	// ResponsesSourceMetadata is the metadata of a document source.
	ResponsesSourceMetadata = openai.ResponsesSourceMetadata
)

// ChatOptions are the Chat Completions options of a call, read from the
// "openai" provider options namespace.
type ChatOptions = openai.ChatOptions
