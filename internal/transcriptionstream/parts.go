package transcriptionstream

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

// Server part types.
const (
	partStreamStart       = "stream-start"
	partTranscriptDelta   = "transcript-delta"
	partTranscriptPartial = "transcript-partial"
	partTranscriptFinal   = "transcript-final"
	partResponseMetadata  = "response-metadata"
	partFinish            = "finish"
	partRaw               = "raw"
	partError             = "error"
)

// RemoteError is an error part received from the server.
type RemoteError struct {
	// Name and Message are set when the server sent an {name, message}
	// object.
	Name    string
	Message string

	// Value is the error payload as sent.
	Value json.RawMessage
}

// Error returns the message, or the raw payload when there is none.
func (e *RemoteError) Error() string {
	if e.Message != "" {
		return e.Message
	}

	return string(e.Value)
}

// metadataWire is provider metadata with each namespace kept as JSON.
type metadataWire map[string]json.RawMessage

type partHeader struct {
	Type string `json:"type"`
}

type warningWire struct {
	Type    string `json:"type"`
	Feature string `json:"feature,omitempty"`
	Details string `json:"details,omitempty"`
	Setting string `json:"setting,omitempty"`
	Message string `json:"message,omitempty"`
}

type streamStartWire struct {
	Type     string        `json:"type"`
	Warnings []warningWire `json:"warnings"`
}

type transcriptDeltaWire struct {
	Type             string       `json:"type"`
	ID               string       `json:"id,omitempty"`
	Delta            *string      `json:"delta"`
	ProviderMetadata metadataWire `json:"providerMetadata,omitempty"`
}

type transcriptPartialWire struct {
	Type              string       `json:"type"`
	ID                string       `json:"id,omitempty"`
	Text              *string      `json:"text"`
	StartSecond       *float64     `json:"startSecond,omitempty"`
	DurationInSeconds *float64     `json:"durationInSeconds,omitempty"`
	ChannelIndex      *int         `json:"channelIndex,omitempty"`
	ProviderMetadata  metadataWire `json:"providerMetadata,omitempty"`
}

type transcriptFinalWire struct {
	Type             string       `json:"type"`
	ID               string       `json:"id,omitempty"`
	Text             *string      `json:"text"`
	StartSecond      *float64     `json:"startSecond,omitempty"`
	EndSecond        *float64     `json:"endSecond,omitempty"`
	ChannelIndex     *int         `json:"channelIndex,omitempty"`
	ProviderMetadata metadataWire `json:"providerMetadata,omitempty"`
}

type responseMetadataWire struct {
	Type      string            `json:"type"`
	Timestamp *string           `json:"timestamp,omitempty"`
	ModelID   string            `json:"modelId,omitempty"`
	Headers   map[string]string `json:"headers,omitempty"`
}

type segmentWire struct {
	Text        *string  `json:"text"`
	StartSecond *float64 `json:"startSecond"`
	EndSecond   *float64 `json:"endSecond"`
}

type finishWire struct {
	Type              string          `json:"type"`
	Text              *string         `json:"text"`
	Segments          []segmentWire   `json:"segments"`
	Language          string          `json:"language,omitempty"`
	DurationInSeconds *float64        `json:"durationInSeconds,omitempty"`
	Usage             json.RawMessage `json:"usage,omitempty"`
	ProviderMetadata  metadataWire    `json:"providerMetadata,omitempty"`
}

type rawWire struct {
	Type     string          `json:"type"`
	RawValue json.RawMessage `json:"rawValue"`
}

type errorWire struct {
	Type  string          `json:"type"`
	Error json.RawMessage `json:"error"`
}

type errorValueWire struct {
	Name    string `json:"name"`
	Message string `json:"message"`
}

// EncodePart serializes a part as one server TEXT frame. Errors become
// {name, message} objects.
func EncodePart(part model.TranscriptionStreamPart) ([]byte, error) {
	wire, err := partToWire(part)
	if err != nil {
		return nil, err
	}

	encoded, err := json.Marshal(wire)
	if err != nil {
		return nil, fmt.Errorf("encode transcription stream part: %w", err)
	}

	return encoded, nil
}

// partToWire returns the wire struct of a part.
func partToWire(part model.TranscriptionStreamPart) (any, error) {
	switch typed := part.(type) {
	case model.StreamStart:
		return streamStartToWire(typed), nil

	case model.TranscriptDelta:
		return transcriptDeltaToWire(typed)

	case model.TranscriptPartial:
		return transcriptPartialToWire(typed)

	case model.TranscriptFinal:
		return transcriptFinalToWire(typed)

	case model.ResponseMetadata:
		return responseMetadataToWire(typed), nil

	case model.TranscriptFinish:
		return finishToWire(typed)

	case model.Raw:
		return rawWire{
			Type:     partRaw,
			RawValue: typed.Value,
		}, nil

	case model.StreamError:
		return errorToWire(typed.Err)

	default:
		return nil, fmt.Errorf("%w: unknown transcription stream part %T", model.ErrInvalidArgument, part)
	}
}

func streamStartToWire(part model.StreamStart) streamStartWire {
	warnings := make([]warningWire, 0, len(part.Warnings))
	for _, warning := range part.Warnings {
		warnings = append(warnings, warningWire{
			Type:    string(warning.Type),
			Feature: warning.Feature,
			Details: warning.Details,
			Setting: warning.Setting,
			Message: warning.Message,
		})
	}

	return streamStartWire{
		Type:     partStreamStart,
		Warnings: warnings,
	}
}

func transcriptDeltaToWire(part model.TranscriptDelta) (transcriptDeltaWire, error) {
	metadata, err := metadataToWire(part.ProviderMetadata)
	if err != nil {
		return transcriptDeltaWire{}, err
	}

	return transcriptDeltaWire{
		Type:             partTranscriptDelta,
		ID:               part.ID,
		Delta:            new(part.Delta),
		ProviderMetadata: metadata,
	}, nil
}

func transcriptPartialToWire(part model.TranscriptPartial) (transcriptPartialWire, error) {
	metadata, err := metadataToWire(part.ProviderMetadata)
	if err != nil {
		return transcriptPartialWire{}, err
	}

	return transcriptPartialWire{
		Type:              partTranscriptPartial,
		ID:                part.ID,
		Text:              new(part.Text),
		StartSecond:       secondsOf(part.Start),
		DurationInSeconds: secondsOf(part.Duration),
		ChannelIndex:      part.ChannelIndex,
		ProviderMetadata:  metadata,
	}, nil
}

func transcriptFinalToWire(part model.TranscriptFinal) (transcriptFinalWire, error) {
	metadata, err := metadataToWire(part.ProviderMetadata)
	if err != nil {
		return transcriptFinalWire{}, err
	}

	return transcriptFinalWire{
		Type:             partTranscriptFinal,
		ID:               part.ID,
		Text:             new(part.Text),
		StartSecond:      secondsOf(part.Start),
		EndSecond:        secondsOf(part.End),
		ChannelIndex:     part.ChannelIndex,
		ProviderMetadata: metadata,
	}, nil
}

func responseMetadataToWire(part model.ResponseMetadata) responseMetadataWire {
	wire := responseMetadataWire{
		Type:    partResponseMetadata,
		ModelID: part.ModelID,
	}

	if !part.Timestamp.IsZero() {
		wire.Timestamp = new(part.Timestamp.UTC().Format(time.RFC3339Nano))
	}

	return wire
}

func finishToWire(part model.TranscriptFinish) (finishWire, error) {
	metadata, err := metadataToWire(part.ProviderMetadata)
	if err != nil {
		return finishWire{}, err
	}

	segments := make([]segmentWire, 0, len(part.Segments))
	for _, segment := range part.Segments {
		segments = append(segments, segmentWire{
			Text:        new(segment.Text),
			StartSecond: new(segment.Start.Seconds()),
			EndSecond:   new(segment.End.Seconds()),
		})
	}

	wire := finishWire{
		Type:             partFinish,
		Text:             new(part.Text),
		Segments:         segments,
		Language:         part.Language,
		Usage:            part.Usage,
		ProviderMetadata: metadata,
	}

	if part.Duration != 0 {
		wire.DurationInSeconds = new(part.Duration.Seconds())
	}

	return wire, nil
}

// errorToWire keeps the name and message of a received RemoteError.
func errorToWire(err error) (errorWire, error) {
	value := errorValueWire{
		Name:    "Error",
		Message: "unknown error",
	}

	if err != nil {
		value.Message = err.Error()
	}

	var remote *RemoteError
	if errors.As(err, &remote) && remote.Name != "" {
		value.Name = remote.Name
	}

	encoded, marshalErr := json.Marshal(value)
	if marshalErr != nil {
		return errorWire{}, fmt.Errorf("encode transcription stream error: %w", marshalErr)
	}

	return errorWire{
		Type:  partError,
		Error: encoded,
	}, nil
}

// metadataToWire encodes each provider metadata namespace as JSON.
func metadataToWire(metadata model.ProviderMetadata) (metadataWire, error) {
	if len(metadata) == 0 {
		return nil, nil
	}

	wire := make(metadataWire, len(metadata))
	for namespace, value := range metadata {
		encoded, err := json.Marshal(value)
		if err != nil {
			return nil, fmt.Errorf("encode %s provider metadata: %w", namespace, err)
		}

		wire[namespace] = encoded
	}

	return wire, nil
}

func secondsOf(duration *time.Duration) *float64 {
	if duration == nil {
		return nil
	}

	return new(duration.Seconds())
}
