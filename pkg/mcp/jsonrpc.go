package mcp

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
)

const jsonRPCVersion = "2.0"

// ID is a JSON-RPC request ID: a string or an integer.
type ID struct {
	text     string
	number   int64
	isString bool
}

// NumberID returns an integer request ID.
func NumberID(number int64) ID {
	return ID{
		number: number,
	}
}

// StringID returns a string request ID.
func StringID(text string) ID {
	return ID{
		text:     text,
		isString: true,
	}
}

// String returns the ID as text.
func (id ID) String() string {
	if id.isString {
		return id.text
	}

	return strconv.FormatInt(id.number, 10)
}

// MarshalJSON encodes the ID as a JSON string or number.
func (id ID) MarshalJSON() ([]byte, error) {
	if id.isString {
		return json.Marshal(id.text)
	}

	return []byte(strconv.FormatInt(id.number, 10)), nil
}

// UnmarshalJSON decodes a JSON string or integer.
func (id *ID) UnmarshalJSON(data []byte) error {
	if len(data) > 0 && data[0] == '"' {
		var text string
		if err := json.Unmarshal(data, &text); err != nil {
			return err
		}

		*id = StringID(text)

		return nil
	}

	number, err := parseInteger(string(data))
	if err != nil {
		return fmt.Errorf("json-rpc id must be a string or an integer: %w", err)
	}

	*id = NumberID(number)

	return nil
}

// correlationNumber returns the numeric value used to match a response with
// a request. Like upstream, numeric strings match numeric IDs.
func (id ID) correlationNumber() (int64, bool) {
	if !id.isString {
		return id.number, true
	}

	number, err := parseInteger(id.text)
	if err != nil {
		return 0, false
	}

	return number, true
}

// parseInteger parses a JSON number that has an integral value, such as 3 or
// 3.0.
func parseInteger(text string) (int64, error) {
	if number, err := strconv.ParseInt(text, 10, 64); err == nil {
		return number, nil
	}

	value, err := strconv.ParseFloat(text, 64)
	if err != nil {
		return 0, err
	}

	if value != math.Trunc(value) || math.Abs(value) > maxSafeInteger {
		return 0, fmt.Errorf("%s is not an integer", text)
	}

	return int64(value), nil
}

// maxSafeInteger is the largest integer a JSON number represents exactly in
// JavaScript, the reference implementation's number type.
const maxSafeInteger = 1<<53 - 1

// RPCError is the error member of a JSON-RPC error response.
type RPCError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

// Error returns the code and message.
func (e *RPCError) Error() string {
	return fmt.Sprintf("json-rpc error %d: %s", e.Code, e.Message)
}

// JSON-RPC error codes the client sends.
const (
	errorCodeInvalidParams  = -32602
	errorCodeMethodNotFound = -32601
	errorCodeInternalError  = -32603
)

// MessageKind classifies a JSON-RPC message.
type MessageKind string

// Message kinds.
const (
	MessageKindRequest      MessageKind = "request"
	MessageKindNotification MessageKind = "notification"
	MessageKindResponse     MessageKind = "response"
	MessageKindError        MessageKind = "error"
)

// Message is a JSON-RPC 2.0 message. Params and Result stay encoded until the
// receiver decodes them into the typed MCP structure for the method.
type Message struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      *ID             `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *RPCError       `json:"error,omitempty"`
}

// Kind classifies a valid message.
func (m Message) Kind() MessageKind {
	switch {
	case m.Method != "" && m.ID != nil:
		return MessageKindRequest
	case m.Method != "":
		return MessageKindNotification
	case m.Error != nil:
		return MessageKindError
	default:
		return MessageKindResponse
	}
}

// ParseMessage decodes and validates one JSON-RPC message. Unknown members
// are rejected, as in upstream's strict message schemas.
func ParseMessage(data []byte) (Message, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()

	var message Message
	if err := decoder.Decode(&message); err != nil {
		return Message{}, fmt.Errorf("decode json-rpc message: %w", err)
	}

	if _, err := decoder.Token(); !errors.Is(err, io.EOF) {
		return Message{}, errors.New("decode json-rpc message: unexpected data after message")
	}

	if err := ValidateMessage(message); err != nil {
		return Message{}, err
	}

	return message, nil
}

// ValidateMessage reports whether message is a valid request, notification,
// response or error response.
func ValidateMessage(message Message) error {
	if message.JSONRPC != jsonRPCVersion {
		return fmt.Errorf("invalid json-rpc version %q", message.JSONRPC)
	}

	switch {
	case message.Method != "":
		return validateRequestMessage(message)
	case message.Error != nil:
		if message.Result != nil {
			return errors.New("json-rpc error response must not have a result")
		}

		return nil
	case message.Result != nil:
		if message.ID == nil {
			return errors.New("json-rpc response must have an id")
		}

		if !isJSONObject(message.Result) {
			return errors.New("json-rpc result must be an object")
		}

		return nil
	default:
		return errors.New("json-rpc message must have a method, a result or an error")
	}
}

func validateRequestMessage(message Message) error {
	if message.Result != nil || message.Error != nil {
		return errors.New("json-rpc request must not have a result or an error")
	}

	if message.Params != nil && !isJSONObject(message.Params) {
		return errors.New("json-rpc params must be an object")
	}

	return nil
}

// isJSONObject reports whether raw, which is valid JSON, encodes an object.
func isJSONObject(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)

	return len(trimmed) > 0 && trimmed[0] == '{'
}

func newRequestMessage(id ID, method string, params json.RawMessage) Message {
	return Message{
		JSONRPC: jsonRPCVersion,
		ID:      &id,
		Method:  method,
		Params:  params,
	}
}

func newNotificationMessage(method string, params json.RawMessage) Message {
	return Message{
		JSONRPC: jsonRPCVersion,
		Method:  method,
		Params:  params,
	}
}

func newResultMessage(id ID, result json.RawMessage) Message {
	return Message{
		JSONRPC: jsonRPCVersion,
		ID:      &id,
		Result:  result,
	}
}

func newErrorMessage(id ID, rpcError RPCError) Message {
	return Message{
		JSONRPC: jsonRPCVersion,
		ID:      &id,
		Error:   &rpcError,
	}
}
