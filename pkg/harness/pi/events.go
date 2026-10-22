package pi

import (
	"encoding/json"
	"strings"
)

// sessionEvent is the part of Pi's RPC session events the adapter reads,
// the Go form of upstream's loose piSessionEventSchema. RPC message_update
// records are delta-only: they carry no partial message, and toolcall_start
// names the tool call directly.
type sessionEvent struct {
	Type                  string                 `json:"type"`
	AssistantMessageEvent *assistantMessageEvent `json:"assistantMessageEvent"`

	ToolCallID string          `json:"toolCallId"`
	ToolName   string          `json:"toolName"`
	Args       json.RawMessage `json:"args"`
	Result     json.RawMessage `json:"result"`
	IsError    bool            `json:"isError"`

	// Reason and Aborted describe compaction events; Result carries the
	// compaction result.
	Reason  string `json:"reason"`
	Aborted bool   `json:"aborted"`

	Message *sessionMessage `json:"message"`
}

// assistantMessageEvent is one content-block update of an assistant
// message.
type assistantMessageEvent struct {
	Type         string `json:"type"`
	ContentIndex *int   `json:"contentIndex"`
	Delta        string `json:"delta"`
	ID           string `json:"id"`
	ToolName     string `json:"toolName"`
}

// sessionMessage is the part of a Pi message the adapter reads.
type sessionMessage struct {
	Role         string          `json:"role"`
	Content      json.RawMessage `json:"content"`
	StopReason   string          `json:"stopReason"`
	ErrorMessage string          `json:"errorMessage"`
	Usage        *messageUsage   `json:"usage"`
}

// messageUsage is Pi's usage of one assistant message. Input excludes cached
// tokens; reasoning is a subset of output, set only by providers that report
// it.
type messageUsage struct {
	Input      int  `json:"input"`
	Output     int  `json:"output"`
	CacheRead  int  `json:"cacheRead"`
	CacheWrite int  `json:"cacheWrite"`
	Reasoning  *int `json:"reasoning,omitempty"`
}

// contentBlock is one block of a message's content.
type contentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
	ID   string `json:"id"`
	Name string `json:"name"`
}

// blocks decodes the content blocks of a message, or nil for string
// content.
func (m *sessionMessage) blocks() []contentBlock {
	if m == nil {
		return nil
	}

	var blocks []contentBlock
	if err := json.Unmarshal(m.Content, &blocks); err != nil {
		return nil
	}

	return blocks
}

// assistantText returns the text of an assistant message, as upstream
// extractAssistantText.
func assistantText(message *sessionMessage) string {
	if message == nil || message.Role != "assistant" {
		return ""
	}

	var text string
	if err := json.Unmarshal(message.Content, &text); err == nil {
		return text
	}

	var joined strings.Builder
	for _, block := range message.blocks() {
		if block.Type == "text" {
			joined.WriteString(block.Text)
		}
	}

	return joined.String()
}

// toolCallIDs returns the tool call IDs of an assistant message.
func toolCallIDs(message *sessionMessage) []string {
	if message == nil || message.Role != "assistant" {
		return nil
	}

	var ids []string
	for _, block := range message.blocks() {
		if block.Type == "toolCall" && block.ID != "" {
			ids = append(ids, block.ID)
		}
	}

	return ids
}

// terminalError returns the error of an assistant message that ended with
// an error or an abort, as upstream getPiTerminalError.
func terminalError(event sessionEvent) string {
	message := event.Message
	if message == nil {
		return ""
	}

	if text := strings.TrimSpace(message.ErrorMessage); text != "" {
		return text
	}

	stopReason := strings.TrimSpace(message.StopReason)
	if stopReason == "error" || stopReason == "aborted" {
		return stopReason
	}

	return ""
}
