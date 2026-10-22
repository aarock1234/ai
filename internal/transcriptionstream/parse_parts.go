package transcriptionstream

import (
	"time"

	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/model"
)

// ParsePart parses a server TEXT frame on the client side. It reports
// false for malformed JSON, unknown part types, and known parts whose
// fields are missing or mistyped, so a drifted server cannot produce parts
// that downstream code misreads. Response metadata timestamps are parsed
// from ISO 8601 strings.
func ParsePart(text []byte) (model.TranscriptionStreamPart, bool) {
	var header partHeader
	if !decodeObject(text, &header) {
		return nil, false
	}

	switch header.Type {
	case partStreamStart:
		return parseStreamStart(text)

	case partTranscriptDelta:
		return parseTranscriptDelta(text)

	case partTranscriptPartial:
		return parseTranscriptPartial(text)

	case partTranscriptFinal:
		return parseTranscriptFinal(text)

	case partResponseMetadata:
		return parseResponseMetadata(text)

	case partFinish:
		return parseFinish(text)

	case partRaw:
		return parseRaw(text)

	case partError:
		return parseError(text)

	default:
		return nil, false
	}
}

func parseStreamStart(text []byte) (model.TranscriptionStreamPart, bool) {
	var wire streamStartWire
	if !decodeObject(text, &wire) || wire.Warnings == nil {
		return nil, false
	}

	warnings := make([]model.Warning, 0, len(wire.Warnings))
	for _, warning := range wire.Warnings {
		if warning.Type == "" {
			return nil, false
		}

		warnings = append(warnings, model.Warning{
			Type:    model.WarningType(warning.Type),
			Feature: warning.Feature,
			Details: warning.Details,
			Setting: warning.Setting,
			Message: warning.Message,
		})
	}

	return model.StreamStart{
		Warnings: warnings,
	}, true
}

func parseTranscriptDelta(text []byte) (model.TranscriptionStreamPart, bool) {
	var wire transcriptDeltaWire
	if !decodeObject(text, &wire) || wire.Delta == nil {
		return nil, false
	}

	return model.TranscriptDelta{
		ID:               wire.ID,
		Delta:            *wire.Delta,
		ProviderMetadata: wire.ProviderMetadata.metadata(),
	}, true
}

func parseTranscriptPartial(text []byte) (model.TranscriptionStreamPart, bool) {
	var wire transcriptPartialWire
	if !decodeObject(text, &wire) || wire.Text == nil {
		return nil, false
	}

	return model.TranscriptPartial{
		ID:               wire.ID,
		Text:             *wire.Text,
		Start:            durationOf(wire.StartSecond),
		Duration:         durationOf(wire.DurationInSeconds),
		ChannelIndex:     wire.ChannelIndex,
		ProviderMetadata: wire.ProviderMetadata.metadata(),
	}, true
}

func parseTranscriptFinal(text []byte) (model.TranscriptionStreamPart, bool) {
	var wire transcriptFinalWire
	if !decodeObject(text, &wire) || wire.Text == nil {
		return nil, false
	}

	return model.TranscriptFinal{
		ID:               wire.ID,
		Text:             *wire.Text,
		Start:            durationOf(wire.StartSecond),
		End:              durationOf(wire.EndSecond),
		ChannelIndex:     wire.ChannelIndex,
		ProviderMetadata: wire.ProviderMetadata.metadata(),
	}, true
}

// parseResponseMetadata reads the part. Headers are validated but not
// kept: model.ResponseMetadata has no headers.
func parseResponseMetadata(text []byte) (model.TranscriptionStreamPart, bool) {
	var wire responseMetadataWire
	if !decodeObject(text, &wire) {
		return nil, false
	}

	metadata := model.ResponseMetadata{
		ModelID: wire.ModelID,
	}

	if wire.Timestamp != nil {
		timestamp, err := time.Parse(time.RFC3339Nano, *wire.Timestamp)
		if err != nil {
			return nil, false
		}

		metadata.Timestamp = timestamp
	}

	return metadata, true
}

func parseFinish(text []byte) (model.TranscriptionStreamPart, bool) {
	var wire finishWire
	if !decodeObject(text, &wire) {
		return nil, false
	}

	hasRequiredFields := wire.Text != nil && wire.Segments != nil
	hasValidUsage := !wirejson.IsSet(wire.Usage) || isJSONObject(wire.Usage)

	if !hasRequiredFields || !hasValidUsage {
		return nil, false
	}

	segments := make([]model.TranscriptionSegment, 0, len(wire.Segments))
	for _, segment := range wire.Segments {
		isComplete := segment.Text != nil && segment.StartSecond != nil && segment.EndSecond != nil
		if !isComplete {
			return nil, false
		}

		segments = append(segments, model.TranscriptionSegment{
			Text:  *segment.Text,
			Start: secondsToDuration(*segment.StartSecond),
			End:   secondsToDuration(*segment.EndSecond),
		})
	}

	finish := model.TranscriptFinish{
		Text:             *wire.Text,
		Segments:         segments,
		Language:         wire.Language,
		Usage:            wire.Usage,
		ProviderMetadata: wire.ProviderMetadata.metadata(),
	}

	if wire.DurationInSeconds != nil {
		finish.Duration = secondsToDuration(*wire.DurationInSeconds)
	}

	return finish, true
}

func parseRaw(text []byte) (model.TranscriptionStreamPart, bool) {
	var wire rawWire
	if !decodeObject(text, &wire) || wire.RawValue == nil {
		return nil, false
	}

	return model.Raw{
		Value: wire.RawValue,
	}, true
}

// parseError reads an error part. {name, message} objects keep their
// fields; any other payload is kept as sent.
func parseError(text []byte) (model.TranscriptionStreamPart, bool) {
	var wire errorWire
	if !decodeObject(text, &wire) || wire.Error == nil {
		return nil, false
	}

	remote := &RemoteError{
		Value: wire.Error,
	}

	var value errorValueWire
	if decodeObject(wire.Error, &value) {
		remote.Name = value.Name
		remote.Message = value.Message
	}

	return model.StreamError{
		Err: remote,
	}, true
}

// metadata returns the provider metadata, with each namespace as raw JSON
// that model.ProviderValue decodes.
func (w metadataWire) metadata() model.ProviderMetadata {
	if len(w) == 0 {
		return nil
	}

	metadata := make(model.ProviderMetadata, len(w))
	for namespace, value := range w {
		metadata[namespace] = value
	}

	return metadata
}

func durationOf(seconds *float64) *time.Duration {
	if seconds == nil {
		return nil
	}

	return new(secondsToDuration(*seconds))
}

func secondsToDuration(seconds float64) time.Duration {
	return time.Duration(seconds * float64(time.Second))
}
