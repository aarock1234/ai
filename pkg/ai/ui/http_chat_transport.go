package ui

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"maps"
	"net/http"
	"net/url"
	"strings"
	"unicode/utf8"

	"github.com/aarock1234/ai/pkg/model"
)

// chatErrorFallback is the message of a failed chat request without a
// response body.
const chatErrorFallback = "Failed to fetch the chat response."

// maxChatErrorBodyBytes bounds the error body read from a failed request.
const maxChatErrorBodyBytes = 1 << 20

// chatStreamProtocol is the encoding of chat responses.
type chatStreamProtocol int

const (
	// uiMessageStreamProtocol is the UI message stream in Server-Sent
	// Events.
	uiMessageStreamProtocol chatStreamProtocol = iota

	// textStreamProtocol is plain text.
	textStreamProtocol
)

// PrepareSendMessagesRequestOptions are passed to
// HTTPChatTransportOptions.PrepareSendMessagesRequest.
type PrepareSendMessagesRequestOptions struct {
	API string

	// ID is the chat ID.
	ID string

	Messages        []Message
	RequestMetadata json.RawMessage

	// Body merges the transport's and the request's body members.
	Body map[string]json.RawMessage

	// Headers merges the transport's and the request's headers.
	Headers http.Header

	Trigger   ChatTrigger
	MessageID string
}

// PreparedSendMessagesRequest replaces parts of the default request.
type PreparedSendMessagesRequest struct {
	// Body replaces the default body, encoded with encoding/json, when it is
	// not nil.
	Body any

	// Headers replace the merged headers when they are not nil.
	Headers http.Header

	// API replaces the endpoint when it is not empty.
	API string
}

// PrepareReconnectToStreamRequestOptions are passed to
// HTTPChatTransportOptions.PrepareReconnectToStreamRequest.
type PrepareReconnectToStreamRequestOptions struct {
	API string

	// ID is the chat ID.
	ID string

	RequestMetadata json.RawMessage
	Body            map[string]json.RawMessage
	Headers         http.Header
}

// PreparedReconnectToStreamRequest replaces parts of the default reconnect
// request.
type PreparedReconnectToStreamRequest struct {
	// Headers replace the merged headers when they are not nil.
	Headers http.Header

	// API replaces the default URL, the endpoint followed by
	// "/{chatId}/stream", when it is not empty.
	API string
}

// HTTPChatTransportOptions configure the HTTP chat transports.
//
// Go adaptations: headers and body are values; compute them per request in
// the preparation hooks, which receive the request's context. Browser
// credentials have no Go equivalent; set authentication headers instead.
type HTTPChatTransportOptions struct {
	// API is the absolute URL of the chat endpoint. It is required.
	API string

	Headers http.Header

	// Body members are added to every request body.
	Body map[string]json.RawMessage

	// HTTPClient sends the requests. Nil uses http.DefaultClient.
	HTTPClient *http.Client

	// PrepareSendMessagesRequest optionally replaces the endpoint, headers
	// or body of a send request. A nil result keeps the defaults.
	PrepareSendMessagesRequest func(ctx context.Context, options PrepareSendMessagesRequestOptions) (*PreparedSendMessagesRequest, error)

	// PrepareReconnectToStreamRequest optionally replaces the URL or headers
	// of a reconnect request. A nil result keeps the defaults.
	PrepareReconnectToStreamRequest func(ctx context.Context, options PrepareReconnectToStreamRequestOptions) (*PreparedReconnectToStreamRequest, error)
}

// DefaultChatTransportOptions configure NewDefaultChatTransport.
type DefaultChatTransportOptions struct {
	HTTPChatTransportOptions

	// ResumeStreamIsReplay reports that the backend replays a resumed
	// stream from its start. Nil means true.
	ResumeStreamIsReplay *bool
}

// HTTPChatTransport sends chat requests to an HTTP endpoint: the
// conversation as a JSON POST, and reconnects as a GET of
// "{api}/{chatId}/stream", which answers 204 without an active stream.
type HTTPChatTransport struct {
	options              HTTPChatTransportOptions
	protocol             chatStreamProtocol
	resumeStreamIsReplay bool
}

var _ ChatTransport = (*HTTPChatTransport)(nil)

// NewDefaultChatTransport returns a transport whose responses are UI
// message streams, upstream's DefaultChatTransport.
func NewDefaultChatTransport(options DefaultChatTransportOptions) *HTTPChatTransport {
	resumeStreamIsReplay := true
	if options.ResumeStreamIsReplay != nil {
		resumeStreamIsReplay = *options.ResumeStreamIsReplay
	}

	return &HTTPChatTransport{
		options:              options.HTTPChatTransportOptions,
		protocol:             uiMessageStreamProtocol,
		resumeStreamIsReplay: resumeStreamIsReplay,
	}
}

// NewTextStreamChatTransport returns a transport whose responses are plain
// text, each converted into a message with one text part, upstream's
// TextStreamChatTransport.
func NewTextStreamChatTransport(options HTTPChatTransportOptions) *HTTPChatTransport {
	return &HTTPChatTransport{
		options:  options,
		protocol: textStreamProtocol,
	}
}

// ResumeStreamIsReplay reports whether a resumed stream starts over, so
// that the client must discard what it received before reconnecting.
func (t *HTTPChatTransport) ResumeStreamIsReplay() bool {
	return t.resumeStreamIsReplay
}

// SendMessages posts the conversation and returns the response stream.
func (t *HTTPChatTransport) SendMessages(ctx context.Context, options SendMessagesOptions) (*ChatStream, error) {
	if t.options.API == "" {
		return nil, errors.New("send chat messages: an api url is required")
	}

	headers := mergeChatHeaders(t.options.Headers, options.Headers)
	body := mergeChatBody(t.options.Body, options.Body)

	prepared, err := t.prepareSend(ctx, PrepareSendMessagesRequestOptions{
		API:             t.options.API,
		ID:              options.ChatID,
		Messages:        options.Messages,
		RequestMetadata: options.Metadata,
		Body:            body,
		Headers:         headers,
		Trigger:         options.Trigger,
		MessageID:       options.MessageID,
	})
	if err != nil {
		return nil, err
	}

	endpoint := t.options.API
	if prepared.API != "" {
		endpoint = prepared.API
	}

	if prepared.Headers != nil {
		headers = prepared.Headers.Clone()
	}

	requestBody, err := sendMessagesBody(prepared.Body, body, options)
	if err != nil {
		return nil, err
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(requestBody))
	if err != nil {
		return nil, fmt.Errorf("create chat request: %w", err)
	}

	request.Header = withJSONContentType(headers)

	response, err := t.send(request)
	if err != nil {
		return nil, err
	}

	return t.stream(request, response)
}

func (t *HTTPChatTransport) prepareSend(ctx context.Context, options PrepareSendMessagesRequestOptions) (PreparedSendMessagesRequest, error) {
	if t.options.PrepareSendMessagesRequest == nil {
		return PreparedSendMessagesRequest{}, nil
	}

	prepared, err := t.options.PrepareSendMessagesRequest(ctx, options)
	if err != nil || prepared == nil {
		return PreparedSendMessagesRequest{}, err
	}

	return *prepared, nil
}

// ReconnectToStream resumes the chat's active stream, or returns nil when
// the backend answers 204 No Content.
func (t *HTTPChatTransport) ReconnectToStream(ctx context.Context, options ReconnectToStreamOptions) (*ChatStream, error) {
	if t.options.API == "" {
		return nil, errors.New("reconnect to chat stream: an api url is required")
	}

	headers := mergeChatHeaders(t.options.Headers, options.Headers)

	prepared, err := t.prepareReconnect(ctx, PrepareReconnectToStreamRequestOptions{
		API:             t.options.API,
		ID:              options.ChatID,
		RequestMetadata: options.Metadata,
		Body:            mergeChatBody(t.options.Body, options.Body),
		Headers:         headers,
	})
	if err != nil {
		return nil, err
	}

	endpoint := prepared.API
	if endpoint == "" {
		endpoint, err = reconnectURL(t.options.API, options.ChatID)
		if err != nil {
			return nil, err
		}
	}

	if prepared.Headers != nil {
		headers = prepared.Headers.Clone()
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("create chat reconnect request: %w", err)
	}

	request.Header = headers

	response, err := t.send(request)
	if err != nil {
		return nil, err
	}

	if response.StatusCode == http.StatusNoContent {
		_ = response.Body.Close()

		return nil, nil
	}

	return t.stream(request, response)
}

func (t *HTTPChatTransport) prepareReconnect(ctx context.Context, options PrepareReconnectToStreamRequestOptions) (PreparedReconnectToStreamRequest, error) {
	if t.options.PrepareReconnectToStreamRequest == nil {
		return PreparedReconnectToStreamRequest{}, nil
	}

	prepared, err := t.options.PrepareReconnectToStreamRequest(ctx, options)
	if err != nil || prepared == nil {
		return PreparedReconnectToStreamRequest{}, err
	}

	return *prepared, nil
}

// send sends a chat request with the transport's client.
func (t *HTTPChatTransport) send(request *http.Request) (*http.Response, error) {
	client := t.options.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	response, err := client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("send chat request: %w", err)
	}

	return response, nil
}

// stream returns a successful response as a chat stream, and a failed one
// as an API call error.
func (t *HTTPChatTransport) stream(request *http.Request, response *http.Response) (*ChatStream, error) {
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, chatAPICallError(response, request.URL.String())
	}

	return newChatStream(t.parse(response.Body), response.Body.Close), nil
}

func (t *HTTPChatTransport) parse(body io.Reader) iter.Seq2[Chunk, error] {
	if t.protocol == textStreamProtocol {
		return textResponseChunks(body)
	}

	return ParseMessageStream(body)
}

// chatAPICallError reads a failed response into an API call error whose
// message is the response body. The request body is never included,
// because it holds the conversation.
func chatAPICallError(response *http.Response, endpoint string) error {
	defer func() {
		_ = response.Body.Close()
	}()

	content, err := io.ReadAll(io.LimitReader(response.Body, maxChatErrorBodyBytes))

	message := string(content)
	if message == "" {
		message = chatErrorFallback
	}

	return &model.APICallError{
		Message:         message,
		URL:             endpoint,
		StatusCode:      response.StatusCode,
		ResponseHeaders: response.Header,
		ResponseBody:    string(content),
		IsRetryable:     model.RetryableStatus(response.StatusCode),
		Cause:           err,
	}
}

// textResponseChunks converts a plain text body into a UI message stream
// with one text part. A read error ends the stream after the text read so
// far.
func textResponseChunks(body io.Reader) iter.Seq2[Chunk, error] {
	return func(yield func(Chunk, error) bool) {
		var readErr error

		text := func(yieldText func(string) bool) {
			readErr = readUTF8Text(body, yieldText)
		}

		for chunk, err := range TextToMessageStream(text) {
			if !yield(chunk, err) {
				return
			}
		}

		if readErr != nil {
			yield(nil, fmt.Errorf("read chat text stream: %w", readErr))
		}
	}
}

// readUTF8Text yields the text of body as it arrives, holding back a
// character split across reads until it is complete.
func readUTF8Text(body io.Reader, yieldText func(string) bool) error {
	buffer := make([]byte, 32<<10)
	var pending []byte

	for {
		count, err := body.Read(buffer)
		pending = append(pending, buffer[:count]...)

		complete := completeUTF8Prefix(pending)
		if complete > 0 {
			if !yieldText(string(pending[:complete])) {
				return nil
			}

			pending = append(pending[:0], pending[complete:]...)
		}

		if errors.Is(err, io.EOF) {
			if len(pending) > 0 {
				yieldText(string(pending))
			}

			return nil
		}

		if err != nil {
			return err
		}
	}
}

// completeUTF8Prefix returns the length of text without a trailing
// incomplete character.
func completeUTF8Prefix(text []byte) int {
	for back := 1; back <= utf8.UTFMax && back <= len(text); back++ {
		start := len(text) - back
		if !utf8.RuneStart(text[start]) {
			continue
		}

		if utf8.FullRune(text[start:]) {
			return len(text)
		}

		return start
	}

	return len(text)
}

// mergeChatHeaders combines the transport's headers with a request's,
// whose values replace those of the same name.
func mergeChatHeaders(base http.Header, overrides http.Header) http.Header {
	merged := base.Clone()
	if merged == nil {
		merged = http.Header{}
	}

	for name, values := range overrides {
		merged[http.CanonicalHeaderKey(name)] = values
	}

	return merged
}

// mergeChatBody combines the transport's body members with a request's.
func mergeChatBody(base map[string]json.RawMessage, overrides map[string]json.RawMessage) map[string]json.RawMessage {
	merged := maps.Clone(base)
	if merged == nil {
		merged = map[string]json.RawMessage{}
	}

	maps.Copy(merged, overrides)

	return merged
}

func withJSONContentType(headers http.Header) http.Header {
	withContentType := http.Header{
		"Content-Type": {
			"application/json",
		},
	}

	for name, values := range headers {
		withContentType[http.CanonicalHeaderKey(name)] = values
	}

	return withContentType
}

// sendMessagesBody encodes the prepared body, or the default body: the
// merged body members with the chat ID, messages, trigger and message ID.
func sendMessagesBody(prepared any, members map[string]json.RawMessage, options SendMessagesOptions) ([]byte, error) {
	if prepared != nil {
		return encodeChatBody(prepared)
	}

	messages := options.Messages
	if messages == nil {
		messages = []Message{}
	}

	encodedMessages, err := encodeJSON(messages)
	if err != nil {
		return nil, fmt.Errorf("encode chat messages: %w", err)
	}

	body := maps.Clone(members)
	body["id"] = jsonStringValue(options.ChatID)
	body["messages"] = encodedMessages
	body["trigger"] = jsonStringValue(string(options.Trigger))

	if options.MessageID != "" {
		body["messageId"] = jsonStringValue(options.MessageID)
	}

	return encodeChatBody(body)
}

func jsonStringValue(text string) json.RawMessage {
	encoded, _ := json.Marshal(text)

	return encoded
}

func encodeChatBody(body any) ([]byte, error) {
	encoded, err := encodeJSON(body)
	if err != nil {
		return nil, fmt.Errorf("encode chat request: %w", err)
	}

	return encoded, nil
}

// reconnectURL appends "/{chatId}/stream" to the path of api, before its
// query and fragment.
func reconnectURL(api string, chatID string) (string, error) {
	if chatID == "." || chatID == ".." {
		return "", ErrInvalidChatID
	}

	path := "/" + url.PathEscape(chatID) + "/stream"

	base, suffix := api, ""
	if index := strings.IndexAny(api, "?#"); index >= 0 {
		base, suffix = api[:index], api[index:]
	}

	base = strings.TrimSuffix(base, "/")

	return base + path + suffix, nil
}
