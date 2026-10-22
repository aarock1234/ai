package harness

import (
	"github.com/aarock1234/ai/pkg/model"
)

// Message is one message of the conversation history a runtime persisted,
// as returned by HistoryReader. Message holds a model.UserMessage,
// model.AssistantMessage or model.ToolMessage using the provider prompt
// shapes; the ProviderOptions of the message and its parts are unused, as
// upstream omits them from harness messages.
type Message struct {
	Message model.Message

	// At is the time the runtime recorded the message, in the runtime's
	// format. It is empty when unknown.
	At string

	// HarnessMetadata is adapter-namespaced data of the message.
	HarnessMetadata Metadata

	// NativeToolNames maps the tool call IDs of the message's tool calls to
	// the runtime's native tool names when they differ from the tool names.
	NativeToolNames map[string]string
}
