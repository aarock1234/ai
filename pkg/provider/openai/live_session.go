package openai

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

const (
	// liveDefaultVoice is the voice of sessions that choose none.
	liveDefaultVoice = "marin"

	// liveMaxInputMessages bounds the conversation history of a session.
	liveMaxInputMessages = 128

	// liveDefaultSampleRate is the PCM rate of WebSocket sessions that
	// choose no audio format.
	liveDefaultSampleRate = 24000
)

// liveTransportKind is the transport a Live API session is built for.
type liveTransportKind string

const (
	liveWebSocket liveTransportKind = "websocket"
	liveWebRTC    liveTransportKind = "webrtc"
)

// LiveOptions are the Live API settings read from the "openai" provider
// options of the session configuration. Unknown fields are rejected.
type LiveOptions struct {
	// Client restricts the events a WebRTC client may send and receive. It
	// only applies to WebRTC sessions.
	Client *LiveClientOptions `json:"client,omitempty"`

	// Delegation selects client delegation. Type "client" is the only
	// supported delegation; an empty Type sends null, as does a JSON null.
	Delegation *LiveDelegation `json:"delegation,omitempty"`

	// Input is the conversation history, at most 128 messages.
	Input []LiveInputMessage `json:"input,omitempty"`

	Store *bool `json:"store,omitempty"`

	// Voice selects a voice by ID. Set either Voice or the session
	// configuration's voice.
	Voice *LiveVoice `json:"voice,omitempty"`
}

// LiveClientOptions are the client permissions of a WebRTC session.
type LiveClientOptions struct {
	DataChannel LiveDataChannel `json:"dataChannel"`
}

// LiveDataChannel lists the events a WebRTC client may exchange.
type LiveDataChannel struct {
	// AllowedClientEvents is "all" or a list of event types.
	AllowedClientEvents *LiveEventAllowList `json:"allowedClientEvents,omitempty"`

	// AllowedServerEvents is "all" or a list of event selectors.
	AllowedServerEvents *LiveServerEventAllowList `json:"allowedServerEvents,omitempty"`
}

// LiveEventAllowList is either every event or the listed event types.
type LiveEventAllowList struct {
	All    bool
	Events []string
}

// LiveServerEventAllowList is either every server event or the selected
// ones.
type LiveServerEventAllowList struct {
	All       bool
	Selectors []LiveServerEventSelector
}

// LiveServerEventSelector selects server events. ResponseEvent is
// required when Type is "response.event" and forbidden otherwise.
type LiveServerEventSelector struct {
	Type          string `json:"type"`
	ResponseEvent string `json:"responseEvent,omitempty"`
}

// LiveDelegation configures delegation.
type LiveDelegation struct {
	Type string `json:"type"`
}

// LiveInputMessage is one message of the conversation history, with one
// text part: "input_text" for developer and user messages, and "text" or
// "output_text" for assistant messages.
type LiveInputMessage struct {
	Type    string             `json:"type"`
	Role    string             `json:"role"`
	Content []LiveInputContent `json:"content"`
}

// LiveInputContent is the text of a history message.
type LiveInputContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// LiveVoice selects a voice by ID.
type LiveVoice struct {
	ID string `json:"id"`
}

// MarshalJSON encodes "all" or the list of events.
func (l LiveEventAllowList) MarshalJSON() ([]byte, error) {
	if l.All {
		return json.Marshal("all")
	}

	return json.Marshal(l.Events)
}

// UnmarshalJSON decodes "all" or a list of event types.
func (l *LiveEventAllowList) UnmarshalJSON(data []byte) error {
	if isJSONAll(data) {
		*l = LiveEventAllowList{
			All: true,
		}

		return nil
	}

	var events []string
	if err := json.Unmarshal(data, &events); err != nil {
		return fmt.Errorf("allowed client events must be \"all\" or a list of event types: %w", err)
	}

	*l = LiveEventAllowList{
		Events: events,
	}

	return nil
}

// MarshalJSON encodes "all" or the list of selectors.
func (l LiveServerEventAllowList) MarshalJSON() ([]byte, error) {
	if l.All {
		return json.Marshal("all")
	}

	return json.Marshal(l.Selectors)
}

// UnmarshalJSON decodes "all" or a list of selectors.
func (l *LiveServerEventAllowList) UnmarshalJSON(data []byte) error {
	if isJSONAll(data) {
		*l = LiveServerEventAllowList{
			All: true,
		}

		return nil
	}

	var selectors []LiveServerEventSelector
	if err := strictUnmarshal(data, &selectors); err != nil {
		return fmt.Errorf("allowed server events must be \"all\" or a list of selectors: %w", err)
	}

	*l = LiveServerEventAllowList{
		Selectors: selectors,
	}

	return nil
}

func isJSONAll(data []byte) bool {
	var text string

	return json.Unmarshal(data, &text) == nil && text == "all"
}

// strictUnmarshal decodes data and rejects unknown fields.
func strictUnmarshal(data []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()

	return decoder.Decode(target)
}

// liveOptions are the decoded Live options with the delegation as sent.
type liveOptions struct {
	LiveOptions

	// delegation is the delegation payload: JSON null, a delegation
	// object, or nil when unset.
	delegation json.RawMessage
}

// readLiveOptions decodes the "openai" provider options strictly and
// validates them.
func readLiveOptions(providerOptions model.ProviderOptions) (liveOptions, error) {
	raw, found, err := model.ProviderValue[json.RawMessage](providerOptions, Namespace)
	if err != nil || !found {
		return liveOptions{}, err
	}

	var options LiveOptions
	if err := strictUnmarshal(raw, &options); err != nil {
		return liveOptions{}, liveDecodeError("providerOptions.openai", err)
	}

	delegation, err := liveDelegation(raw, options.Delegation)
	if err != nil {
		return liveOptions{}, err
	}

	if err := options.validate(); err != nil {
		return liveOptions{}, err
	}

	return liveOptions{
		LiveOptions: options,
		delegation:  delegation,
	}, nil
}

// liveDelegation returns the delegation payload. Responses delegation is
// not supported.
func liveDelegation(raw json.RawMessage, delegation *LiveDelegation) (json.RawMessage, error) {
	var members struct {
		Delegation json.RawMessage `json:"delegation"`
	}

	if err := json.Unmarshal(raw, &members); err != nil {
		return nil, liveDecodeError("providerOptions.openai", err)
	}

	if bytes.Equal(bytes.TrimSpace(members.Delegation), []byte("null")) {
		return json.RawMessage("null"), nil
	}

	if delegation == nil {
		return nil, nil
	}

	switch delegation.Type {
	case "":
		return json.RawMessage("null"), nil
	case "client":
		return json.Marshal(delegation)
	case "responses":
		return nil, fmt.Errorf("%w: OpenAI Live Responses delegation; only client delegation is supported", model.ErrUnsupported)
	default:
		return nil, liveOptionsError("unknown delegation type %q", delegation.Type)
	}
}

func (o LiveOptions) validate() error {
	if len(o.Input) > liveMaxInputMessages {
		return liveOptionsError("input must contain at most %d messages", liveMaxInputMessages)
	}

	for index, message := range o.Input {
		if !isValidLiveInputMessage(message) {
			return liveOptionsError("input message %d must be a message with one text part valid for its role", index)
		}
	}

	if o.Voice != nil && o.Voice.ID == "" {
		return liveOptionsError("voice.id must not be empty")
	}

	if o.Client != nil && o.Client.DataChannel.AllowedServerEvents != nil {
		for _, selector := range o.Client.DataChannel.AllowedServerEvents.Selectors {
			if (selector.Type == "response.event") != (selector.ResponseEvent != "") {
				return liveOptionsError("responseEvent is required for response.event and forbidden for other event types")
			}
		}
	}

	return nil
}

func isValidLiveInputMessage(message LiveInputMessage) bool {
	if message.Type != "message" || len(message.Content) != 1 {
		return false
	}

	contentType := message.Content[0].Type

	switch message.Role {
	case "developer", "user":
		return contentType == "input_text"
	case "assistant":
		return contentType == "text" || contentType == "output_text"
	default:
		return false
	}
}

func liveOptionsError(format string, arguments ...any) error {
	return fmt.Errorf("%w: OpenAI Live options: %s", model.ErrInvalidArgument, fmt.Sprintf(format, arguments...))
}

// liveDecodeError reports provider options that could not be decoded,
// keeping the decode error as a cause.
func liveDecodeError(target string, err error) error {
	return fmt.Errorf("%w: OpenAI Live options: decode %s: %w", model.ErrInvalidArgument, target, err)
}

// liveSession is the Live API session configuration.
type liveSession struct {
	Model        string             `json:"model"`
	Instructions string             `json:"instructions,omitempty"`
	Audio        liveSessionAudio   `json:"audio"`
	Delegation   json.RawMessage    `json:"delegation,omitempty"`
	Client       *liveClient        `json:"client,omitempty"`
	Input        []LiveInputMessage `json:"input,omitempty"`
	Store        *bool              `json:"store,omitempty"`
}

type liveSessionAudio struct {
	Format *realtimeAudioFormat `json:"format,omitempty"`
	Output liveAudioOutput      `json:"output"`
}

type liveAudioOutput struct {
	// Voice is a voice name, or a voice object from the options.
	Voice json.RawMessage `json:"voice"`
}

type liveClient struct {
	DataChannel liveClientDataChannel `json:"data_channel"`
}

type liveClientDataChannel struct {
	AllowedClientEvents *LiveEventAllowList `json:"allowed_client_events,omitempty"`
	AllowedServerEvents json.RawMessage     `json:"allowed_server_events,omitempty"`
}

type liveServerEventSelector struct {
	Type          string `json:"type"`
	ResponseEvent string `json:"response_event,omitempty"`
}

// buildLiveSession converts a normalized configuration. Live sessions take
// instructions, a voice and one audio format; other settings are rejected.
func buildLiveSession(config model.RealtimeSessionConfig, modelID string, transport liveTransportKind) (json.RawMessage, error) {
	if err := checkLiveSessionSettings(config); err != nil {
		return nil, err
	}

	options, err := readLiveOptions(config.ProviderOptions)
	if err != nil {
		return nil, err
	}

	if options.Client != nil && transport != liveWebRTC {
		return nil, fmt.Errorf("%w: OpenAI Live client permissions outside WebRTC startup", model.ErrUnsupported)
	}

	if options.Voice != nil && config.Voice != "" {
		return nil, fmt.Errorf("%w: choose either voice or providerOptions.openai.voice", model.ErrInvalidArgument)
	}

	format, err := liveAudioFormat(config, transport)
	if err != nil {
		return nil, err
	}

	voice, err := liveVoice(config.Voice, options.Voice)
	if err != nil {
		return nil, err
	}

	client, err := liveClientOf(options.Client)
	if err != nil {
		return nil, err
	}

	encoded, err := json.Marshal(liveSession{
		Model:        modelID,
		Instructions: config.Instructions,
		Audio: liveSessionAudio{
			Format: format,
			Output: liveAudioOutput{
				Voice: voice,
			},
		},
		Delegation: options.delegation,
		Client:     client,
		Input:      options.Input,
		Store:      options.Store,
	})
	if err != nil {
		return nil, fmt.Errorf("encode live session: %w", err)
	}

	return encoded, nil
}

// checkLiveSessionSettings rejects settings the Live API does not take.
func checkLiveSessionSettings(config model.RealtimeSessionConfig) error {
	unsupported := []struct {
		name string
		set  bool
	}{
		{
			name: "outputModalities",
			set:  config.OutputModalities != nil,
		},
		{
			name: "inputAudioTranscription",
			set:  config.InputAudioTranscription != nil,
		},
		{
			name: "outputAudioTranscription",
			set:  config.OutputAudioTranscription != nil,
		},
		{
			name: "turnDetection",
			set:  config.TurnDetection != nil,
		},
		{
			name: "tools",
			set:  config.Tools != nil,
		},
	}

	for _, setting := range unsupported {
		if setting.set {
			return fmt.Errorf("%w: OpenAI Live session setting: %s", model.ErrUnsupported, setting.name)
		}
	}

	return nil
}

// liveAudioFormat validates the audio formats. Input and output share one
// format, which WebSocket sessions send and WebRTC sessions negotiate.
func liveAudioFormat(config model.RealtimeSessionConfig, transport liveTransportKind) (*realtimeAudioFormat, error) {
	input := config.InputAudioFormat
	output := config.OutputAudioFormat

	if transport == liveWebRTC {
		if input != nil || output != nil {
			return nil, fmt.Errorf("%w: fixed audio formats for OpenAI Live WebRTC; audio is negotiated through SDP", model.ErrUnsupported)
		}

		return nil, nil
	}

	for _, format := range []*model.AudioFormat{input, output} {
		if format != nil && !isLiveAudioFormat(*format) {
			return nil, fmt.Errorf(
				"%w: OpenAI Live audio must be audio/pcm at 16000 or 24000 Hz, or audio/pcma or audio/pcmu at 8000 Hz",
				model.ErrInvalidArgument,
			)
		}
	}

	if input != nil && output != nil && *input != *output {
		return nil, fmt.Errorf("%w: OpenAI Live requires the same input and output audio format", model.ErrInvalidArgument)
	}

	switch {
	case input != nil:
		return buildRealtimeFormat(input), nil
	case output != nil:
		return buildRealtimeFormat(output), nil
	default:
		return &realtimeAudioFormat{
			Type: "audio/pcm",
			Rate: liveDefaultSampleRate,
		}, nil
	}
}

func isLiveAudioFormat(format model.AudioFormat) bool {
	switch format.Type {
	case "audio/pcm":
		return format.Rate == 16000 || format.Rate == 24000
	case "audio/pcma", "audio/pcmu":
		return format.Rate == 8000
	default:
		return false
	}
}

// liveVoice returns the voice object of the options, the configured voice
// name, or the default voice.
func liveVoice(configVoice string, optionsVoice *LiveVoice) (json.RawMessage, error) {
	if optionsVoice != nil {
		return json.Marshal(optionsVoice)
	}

	if configVoice != "" {
		return json.Marshal(configVoice)
	}

	return json.Marshal(liveDefaultVoice)
}

func liveClientOf(options *LiveClientOptions) (*liveClient, error) {
	if options == nil {
		return nil, nil
	}

	channel := options.DataChannel
	client := &liveClient{
		DataChannel: liveClientDataChannel{
			AllowedClientEvents: channel.AllowedClientEvents,
		},
	}

	allowed := channel.AllowedServerEvents
	if allowed == nil {
		return client, nil
	}

	if allowed.All {
		client.DataChannel.AllowedServerEvents = json.RawMessage(`"all"`)

		return client, nil
	}

	selectors := make([]liveServerEventSelector, len(allowed.Selectors))
	for index, selector := range allowed.Selectors {
		selectors[index] = liveServerEventSelector(selector)
	}

	encoded, err := json.Marshal(selectors)
	if err != nil {
		return nil, fmt.Errorf("encode allowed server events: %w", err)
	}

	client.DataChannel.AllowedServerEvents = encoded

	return client, nil
}
