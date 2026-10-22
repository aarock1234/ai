package gateway

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// Content types that only appear in results.
const (
	contentToolApprovalRequest = "tool-approval-request"
	contentSource              = "source"
)

type wireText struct {
	Text             string         `json:"text"`
	Citations        []wireCitation `json:"citations"`
	ProviderMetadata wireMetadata   `json:"providerMetadata"`
}

func (t wireText) content() (model.Text, error) {
	return model.Text{
		Text:             t.Text,
		Citations:        decodeCitations(t.Citations),
		ProviderMetadata: t.ProviderMetadata.providerMetadata(),
	}, nil
}

type wireReasoning struct {
	Text             string       `json:"text"`
	ProviderMetadata wireMetadata `json:"providerMetadata"`
}

func (r wireReasoning) content() (model.Reasoning, error) {
	return model.Reasoning{
		Text:             r.Text,
		ProviderMetadata: r.ProviderMetadata.providerMetadata(),
	}, nil
}

type wireCustom struct {
	Kind             string       `json:"kind"`
	ProviderMetadata wireMetadata `json:"providerMetadata"`
}

func (c wireCustom) content() (model.Custom, error) {
	return model.Custom{
		Kind:             c.Kind,
		ProviderMetadata: c.ProviderMetadata.providerMetadata(),
	}, nil
}

type wireFile struct {
	MediaType        string       `json:"mediaType"`
	Data             wireFileData `json:"data"`
	ProviderMetadata wireMetadata `json:"providerMetadata"`
}

func (f wireFile) file() (model.File, error) {
	data, err := decodeFileData(f.Data)
	if err != nil {
		return model.File{}, err
	}

	return model.File{
		MediaType:        f.MediaType,
		Data:             data,
		ProviderMetadata: f.ProviderMetadata.providerMetadata(),
	}, nil
}

func (f wireFile) reasoningFile() (model.ReasoningFile, error) {
	data, err := decodeFileData(f.Data)
	if err != nil {
		return model.ReasoningFile{}, err
	}

	return model.ReasoningFile{
		MediaType:        f.MediaType,
		Data:             data,
		ProviderMetadata: f.ProviderMetadata.providerMetadata(),
	}, nil
}

type wireToolApprovalRequest struct {
	ApprovalID       string       `json:"approvalId"`
	ToolCallID       string       `json:"toolCallId"`
	ProviderMetadata wireMetadata `json:"providerMetadata"`
}

func (r wireToolApprovalRequest) content() (model.ToolApprovalRequest, error) {
	return model.ToolApprovalRequest{
		ApprovalID:       r.ApprovalID,
		ToolCallID:       r.ToolCallID,
		ProviderMetadata: r.ProviderMetadata.providerMetadata(),
	}, nil
}

func (s wireSource) content() (model.Source, error) {
	return s.source(), nil
}

type wireToolCall struct {
	ToolCallID       string       `json:"toolCallId"`
	ToolName         string       `json:"toolName"`
	Input            string       `json:"input"`
	ProviderExecuted bool         `json:"providerExecuted"`
	Dynamic          bool         `json:"dynamic"`
	ProviderMetadata wireMetadata `json:"providerMetadata"`
}

func (c wireToolCall) content() (model.ToolCall, error) {
	return model.ToolCall{
		ToolCallID:       c.ToolCallID,
		ToolName:         c.ToolName,
		Input:            c.Input,
		ProviderExecuted: c.ProviderExecuted,
		Dynamic:          c.Dynamic,
		ProviderMetadata: c.ProviderMetadata.providerMetadata(),
	}, nil
}

type wireToolResult struct {
	ToolCallID       string          `json:"toolCallId"`
	ToolName         string          `json:"toolName"`
	Result           json.RawMessage `json:"result"`
	IsError          bool            `json:"isError"`
	Preliminary      bool            `json:"preliminary"`
	Dynamic          bool            `json:"dynamic"`
	ProviderMetadata wireMetadata    `json:"providerMetadata"`
}

func (r wireToolResult) content() (model.ToolResult, error) {
	if len(r.Result) == 0 || string(r.Result) == "null" {
		return model.ToolResult{}, fmt.Errorf("%w: tool result %q has no result", model.ErrInvalidResponse, r.ToolCallID)
	}

	return model.ToolResult{
		ToolCallID:       r.ToolCallID,
		ToolName:         r.ToolName,
		Result:           r.Result,
		IsError:          r.IsError,
		Preliminary:      r.Preliminary,
		Dynamic:          r.Dynamic,
		ProviderMetadata: r.ProviderMetadata.providerMetadata(),
	}, nil
}

// decodeContents converts the content of a generate result, keeping its
// order.
func decodeContents(items []json.RawMessage) ([]model.Content, error) {
	contents := make([]model.Content, 0, len(items))

	for _, item := range items {
		content, err := decodeContent(item)
		if err != nil {
			return nil, err
		}

		contents = append(contents, content)
	}

	return contents, nil
}

// decodeContent converts one content item by its type.
func decodeContent(raw json.RawMessage) (model.Content, error) {
	header, err := decodeAs[typeHeader](raw)
	if err != nil {
		return nil, err
	}

	switch header.Type {
	case partText:
		return asContent(decodeWith(raw, wireText.content))
	case partReasoning:
		return asContent(decodeWith(raw, wireReasoning.content))
	default:
		return asContent(decodeSharedContent(header.Type, raw))
	}
}

// sharedContent is content that appears both in generate results and as a
// complete stream part.
type sharedContent interface {
	model.Content
	model.StreamPart
}

// decodeSharedContent converts content that is also a stream part.
func decodeSharedContent(contentType string, raw json.RawMessage) (sharedContent, error) {
	switch contentType {
	case partCustom:
		return asShared(decodeWith(raw, wireCustom.content))
	case partFile:
		return asShared(decodeWith(raw, wireFile.file))
	case partReasoningFile:
		return asShared(decodeWith(raw, wireFile.reasoningFile))
	case contentToolApprovalRequest:
		return asShared(decodeWith(raw, wireToolApprovalRequest.content))
	case contentSource:
		return asShared(decodeWith(raw, wireSource.content))
	case partToolCall:
		return asShared(decodeWith(raw, wireToolCall.content))
	case partToolResult:
		return asShared(decodeWith(raw, wireToolResult.content))
	default:
		return nil, fmt.Errorf("%w: unknown content type %q", model.ErrInvalidResponse, contentType)
	}
}

// decodeWith decodes raw into its wire type and converts it.
func decodeWith[Wire any, Value any](raw json.RawMessage, convert func(Wire) (Value, error)) (Value, error) {
	wire, err := decodeAs[Wire](raw)
	if err != nil {
		var zero Value

		return zero, err
	}

	return convert(wire)
}

// asContent returns a decoded value as content, or the decoding error.
func asContent[Value model.Content](value Value, err error) (model.Content, error) {
	if err != nil {
		return nil, err
	}

	return value, nil
}

// asShared returns a decoded value as shared content, or the decoding
// error.
func asShared[Value sharedContent](value Value, err error) (sharedContent, error) {
	if err != nil {
		return nil, err
	}

	return value, nil
}
