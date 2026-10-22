package openai

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// responsesOutputItem is one output item of a response, decoded from the
// response body or from an output item stream event.
type responsesOutputItem interface {
	responsesOutputType() responsesItemType
}

// responsesMessageItem is assistant text. Streamed items omit the content.
type responsesMessageItem struct {
	ID      string                `json:"id"`
	Phase   string                `json:"phase"`
	Content []responsesOutputText `json:"content"`
}

// responsesOutputText is one text part of a message.
type responsesOutputText struct {
	Type        string                `json:"type"`
	Text        string                `json:"text"`
	Logprobs    []ResponsesLogprob    `json:"logprobs"`
	Annotations []ResponsesAnnotation `json:"annotations"`
}

// responsesReasoningOutput is a reasoning item.
type responsesReasoningOutput struct {
	ID               string                 `json:"id"`
	EncryptedContent *string                `json:"encrypted_content"`
	Summary          []responsesSummaryText `json:"summary"`
}

// responsesFunctionCallItem is a function call.
type responsesFunctionCallItem struct {
	ID        string           `json:"id"`
	CallID    string           `json:"call_id"`
	Name      string           `json:"name"`
	Arguments string           `json:"arguments"`
	Async     *bool            `json:"async"`
	Namespace string           `json:"namespace"`
	Caller    *responsesCaller `json:"caller"`
	Status    string           `json:"status"`
}

// responsesProgramItem is a program written by programmatic tool calling.
type responsesProgramItem struct {
	ID          string `json:"id"`
	CallID      string `json:"call_id"`
	Code        string `json:"code"`
	Fingerprint string `json:"fingerprint"`
}

// responsesProgramOutputItem is the result of a hosted program.
type responsesProgramOutputItem struct {
	ID     string `json:"id"`
	CallID string `json:"call_id"`
	Result string `json:"result"`
	Status string `json:"status"`
}

// responsesCustomToolCallItem is a custom tool call.
type responsesCustomToolCallItem struct {
	ID     string `json:"id"`
	CallID string `json:"call_id"`
	Name   string `json:"name"`
	Input  string `json:"input"`
	Async  *bool  `json:"async"`
}

// responsesWebSearchCallItem is a web search. Streamed added events omit the
// action.
type responsesWebSearchCallItem struct {
	ID     string                    `json:"id"`
	Status string                    `json:"status"`
	Action *responsesWebSearchAction `json:"action"`
}

// responsesFileSearchCallItem is a file search.
type responsesFileSearchCallItem struct {
	ID      string                      `json:"id"`
	Queries []string                    `json:"queries"`
	Results []responsesFileSearchResult `json:"results"`
}

// responsesFileSearchResult is one file search match.
type responsesFileSearchResult struct {
	Attributes map[string]json.RawMessage `json:"attributes"`
	FileID     string                     `json:"file_id"`
	Filename   string                     `json:"filename"`
	Score      float64                    `json:"score"`
	Text       string                     `json:"text"`
}

// responsesCodeInterpreterCallItem is code run by the code interpreter.
type responsesCodeInterpreterCallItem struct {
	ID          string                  `json:"id"`
	ContainerID string                  `json:"container_id"`
	Code        *string                 `json:"code"`
	Outputs     []CodeInterpreterOutput `json:"outputs"`
	Status      string                  `json:"status"`
}

// responsesImageGenerationCallItem is a generated image. Streamed added
// events omit the result.
type responsesImageGenerationCallItem struct {
	ID     string `json:"id"`
	Result string `json:"result"`
}

// responsesLocalShellCallItem is a local shell call.
type responsesLocalShellCallItem struct {
	ID     string                    `json:"id"`
	CallID string                    `json:"call_id"`
	Action responsesLocalShellAction `json:"action"`
}

// responsesComputerCallItem is a computer call. Calls without a call ID come
// from the legacy computer use tool.
type responsesComputerCallItem struct {
	ID                  string                    `json:"id"`
	CallID              *string                   `json:"call_id"`
	Status              string                    `json:"status"`
	Action              *responsesComputerAction  `json:"action"`
	Actions             []responsesComputerAction `json:"actions"`
	PendingSafetyChecks []ComputerSafetyCheck     `json:"pending_safety_checks"`
}

// responsesMCPCallItem is a call to a remote MCP tool.
type responsesMCPCallItem struct {
	ID                string          `json:"id"`
	Status            string          `json:"status"`
	Arguments         string          `json:"arguments"`
	Name              string          `json:"name"`
	ServerLabel       string          `json:"server_label"`
	Output            *string         `json:"output"`
	Error             json.RawMessage `json:"error"`
	ApprovalRequestID *string         `json:"approval_request_id"`
}

// responsesMCPListToolsItem lists the tools of an MCP server. It is not
// exposed.
type responsesMCPListToolsItem struct {
	ID string `json:"id"`
}

// responsesMCPApprovalRequestItem asks to approve an MCP tool call.
type responsesMCPApprovalRequestItem struct {
	ID                string  `json:"id"`
	ServerLabel       string  `json:"server_label"`
	Name              string  `json:"name"`
	Arguments         string  `json:"arguments"`
	ApprovalRequestID *string `json:"approval_request_id"`
}

// responsesApplyPatchCallItem is a patch for the application to apply.
type responsesApplyPatchCallItem struct {
	ID        string              `json:"id"`
	CallID    string              `json:"call_id"`
	Status    string              `json:"status"`
	Operation ApplyPatchOperation `json:"operation"`
}

// responsesShellCallItem is a shell call.
type responsesShellCallItem struct {
	ID     string               `json:"id"`
	CallID string               `json:"call_id"`
	Status string               `json:"status"`
	Action responsesShellAction `json:"action"`
}

// responsesShellCallOutputItem is the output of a hosted shell call.
type responsesShellCallOutputItem struct {
	ID              string                      `json:"id"`
	CallID          string                      `json:"call_id"`
	Status          string                      `json:"status"`
	MaxOutputLength *int                        `json:"max_output_length"`
	Output          []responsesShellOutputEntry `json:"output"`
}

// responsesCompactionItem is compacted conversation state.
type responsesCompactionItem struct {
	ID               string `json:"id"`
	EncryptedContent string `json:"encrypted_content"`
}

// responsesToolSearchCallItem is a tool search.
type responsesToolSearchCallItem struct {
	ID        string          `json:"id"`
	Execution string          `json:"execution"`
	CallID    *string         `json:"call_id"`
	Status    string          `json:"status"`
	Arguments json.RawMessage `json:"arguments"`
}

// responsesToolSearchOutputItem lists the tools a tool search loaded.
type responsesToolSearchOutputItem struct {
	ID        string            `json:"id"`
	Execution string            `json:"execution"`
	CallID    *string           `json:"call_id"`
	Status    string            `json:"status"`
	Tools     []json.RawMessage `json:"tools"`
}

func (responsesMessageItem) responsesOutputType() responsesItemType {
	return responsesMessage
}

func (responsesReasoningOutput) responsesOutputType() responsesItemType {
	return responsesReasoningItem
}

func (responsesFunctionCallItem) responsesOutputType() responsesItemType {
	return responsesFunctionCall
}

func (responsesProgramItem) responsesOutputType() responsesItemType {
	return responsesProgram
}

func (responsesProgramOutputItem) responsesOutputType() responsesItemType {
	return responsesProgramOutput
}

func (responsesCustomToolCallItem) responsesOutputType() responsesItemType {
	return responsesCustomToolCall
}

func (responsesWebSearchCallItem) responsesOutputType() responsesItemType {
	return responsesWebSearchCall
}

func (responsesFileSearchCallItem) responsesOutputType() responsesItemType {
	return responsesFileSearchCall
}

func (responsesCodeInterpreterCallItem) responsesOutputType() responsesItemType {
	return responsesCodeInterpreterCall
}

func (responsesImageGenerationCallItem) responsesOutputType() responsesItemType {
	return responsesImageGenerationCall
}

func (responsesLocalShellCallItem) responsesOutputType() responsesItemType {
	return responsesLocalShellCall
}

func (responsesComputerCallItem) responsesOutputType() responsesItemType {
	return responsesComputerCall
}

func (responsesMCPCallItem) responsesOutputType() responsesItemType {
	return responsesMCPCall
}

func (responsesMCPListToolsItem) responsesOutputType() responsesItemType {
	return responsesMCPListTools
}

func (responsesMCPApprovalRequestItem) responsesOutputType() responsesItemType {
	return responsesMCPApprovalRequest
}

func (responsesApplyPatchCallItem) responsesOutputType() responsesItemType {
	return responsesApplyPatchCall
}

func (responsesShellCallItem) responsesOutputType() responsesItemType {
	return responsesShellCall
}

func (responsesShellCallOutputItem) responsesOutputType() responsesItemType {
	return responsesShellCallOutput
}

func (responsesCompactionItem) responsesOutputType() responsesItemType {
	return responsesCompaction
}

func (responsesToolSearchCallItem) responsesOutputType() responsesItemType {
	return responsesToolSearchCall
}

func (responsesToolSearchOutputItem) responsesOutputType() responsesItemType {
	return responsesToolSearchOutput
}

// responsesOutput is the output array of a response. Items of unknown types
// are skipped so that new API item types do not break existing callers. A
// JSON null leaves the output nil.
type responsesOutput []responsesOutputItem

// UnmarshalJSON decodes each known item into its concrete type.
func (o *responsesOutput) UnmarshalJSON(data []byte) error {
	var raws []json.RawMessage
	if err := json.Unmarshal(data, &raws); err != nil {
		return err
	}

	if raws == nil {
		return nil
	}

	items := make(responsesOutput, 0, len(raws))
	for _, raw := range raws {
		item, err := decodeResponsesOutputItem(raw)
		if err != nil {
			return err
		}

		if item != nil {
			items = append(items, item)
		}
	}

	*o = items

	return nil
}

// decodeResponsesOutputItem decodes one output item. It returns nil for
// unknown item types.
func decodeResponsesOutputItem(raw json.RawMessage) (responsesOutputItem, error) {
	var header struct {
		Type responsesItemType `json:"type"`
	}
	if err := json.Unmarshal(raw, &header); err != nil {
		return nil, fmt.Errorf("decode output item: %w", err)
	}

	switch header.Type {
	case responsesMessage:
		return decodeResponsesItem[responsesMessageItem](raw, header.Type)
	case responsesReasoningItem:
		return decodeResponsesItem[responsesReasoningOutput](raw, header.Type)
	case responsesFunctionCall:
		return decodeResponsesItem[responsesFunctionCallItem](raw, header.Type)
	case responsesProgram:
		return decodeResponsesItem[responsesProgramItem](raw, header.Type)
	case responsesProgramOutput:
		return decodeResponsesItem[responsesProgramOutputItem](raw, header.Type)
	case responsesCustomToolCall:
		return decodeResponsesItem[responsesCustomToolCallItem](raw, header.Type)
	case responsesWebSearchCall:
		return decodeResponsesItem[responsesWebSearchCallItem](raw, header.Type)
	case responsesFileSearchCall:
		return decodeResponsesItem[responsesFileSearchCallItem](raw, header.Type)
	case responsesCodeInterpreterCall:
		return decodeResponsesItem[responsesCodeInterpreterCallItem](raw, header.Type)
	case responsesImageGenerationCall:
		return decodeResponsesItem[responsesImageGenerationCallItem](raw, header.Type)
	case responsesLocalShellCall:
		return decodeResponsesItem[responsesLocalShellCallItem](raw, header.Type)
	case responsesComputerCall:
		return decodeResponsesItem[responsesComputerCallItem](raw, header.Type)
	case responsesMCPCall:
		return decodeResponsesItem[responsesMCPCallItem](raw, header.Type)
	case responsesMCPListTools:
		return decodeResponsesItem[responsesMCPListToolsItem](raw, header.Type)
	case responsesMCPApprovalRequest:
		return decodeResponsesItem[responsesMCPApprovalRequestItem](raw, header.Type)
	case responsesApplyPatchCall:
		return decodeResponsesItem[responsesApplyPatchCallItem](raw, header.Type)
	case responsesShellCall:
		return decodeResponsesItem[responsesShellCallItem](raw, header.Type)
	case responsesShellCallOutput:
		return decodeResponsesItem[responsesShellCallOutputItem](raw, header.Type)
	case responsesCompaction:
		return decodeResponsesItem[responsesCompactionItem](raw, header.Type)
	case responsesToolSearchCall:
		return decodeResponsesItem[responsesToolSearchCallItem](raw, header.Type)
	case responsesToolSearchOutput:
		return decodeResponsesItem[responsesToolSearchOutputItem](raw, header.Type)
	default:
		return nil, nil
	}
}

// decodeResponsesItem decodes raw into the concrete item type T.
func decodeResponsesItem[T responsesOutputItem](raw json.RawMessage, itemType responsesItemType) (responsesOutputItem, error) {
	var item T
	if err := json.Unmarshal(raw, &item); err != nil {
		return nil, fmt.Errorf("%w: decode %s item: %w", model.ErrInvalidResponse, itemType, err)
	}

	return item, nil
}
