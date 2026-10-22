package harness

import (
	"bytes"
	"encoding/json"
	"strings"
)

// noPrecedingByte marks the start of a value, which is a path boundary.
const noPrecedingByte = -1

// The working directory is stripped from consumer-facing events only.
// Adapters run the agent in a per-session directory under the sandbox root
// and their tools use absolute paths; those paths are correct but noisy, so
// a reference followed by a separator becomes workspace-relative
// ("/work/dir/src/a.go" becomes "src/a.go") and a bare reference becomes
// ".". Boundary-aware replacement is used because tool results are
// free-form text, such as command output, where paths can appear anywhere.

// stripWorkDir returns part with the working directory removed from its
// path-bearing fields.
func stripWorkDir(part StreamPart, workDir string) StreamPart {
	if workDir == "" {
		return part
	}

	switch typed := part.(type) {
	case ToolInputDeltaPart:
		typed.Delta = stripWorkDirString(typed.Delta, workDir)

		return typed
	case ToolCallPart:
		typed.Input = stripWorkDirString(typed.Input, workDir)

		return typed
	case ToolResultPart:
		typed.Result = stripWorkDirJSON(typed.Result, workDir)

		return typed
	case FileChangePart:
		typed.Path = stripWorkDirString(typed.Path, workDir)

		return typed
	default:
		return part
	}
}

// stripWorkDirJSON strips the working directory from every string of a JSON
// value. Invalid JSON is returned unchanged.
func stripWorkDirJSON(value json.RawMessage, workDir string) json.RawMessage {
	if workDir == "" || len(value) == 0 {
		return value
	}

	decoded, ok := decodeJSONValue(value)
	if !ok {
		return value
	}

	encoded, err := json.Marshal(stripWorkDirValue(decoded, workDir))
	if err != nil {
		return value
	}

	return encoded
}

// stripWorkDirOutput strips the working directory from a tool output for
// display. The output is projected through JSON, so the result holds JSON
// values instead of the tool's Go types; outputs that do not encode are
// returned unchanged.
func stripWorkDirOutput(output any, workDir string) any {
	if workDir == "" {
		return output
	}

	if text, ok := output.(string); ok {
		return stripWorkDirString(text, workDir)
	}

	encoded, err := json.Marshal(output)
	if err != nil {
		return output
	}

	decoded, ok := decodeJSONValue(encoded)
	if !ok {
		return output
	}

	return stripWorkDirValue(decoded, workDir)
}

// stripWorkDirValue strips the working directory from every string of a
// decoded JSON value.
func stripWorkDirValue(value any, workDir string) any {
	switch typed := value.(type) {
	case string:
		return stripWorkDirString(typed, workDir)
	case []any:
		stripped := make([]any, len(typed))
		for index, item := range typed {
			stripped[index] = stripWorkDirValue(item, workDir)
		}

		return stripped
	case map[string]any:
		stripped := make(map[string]any, len(typed))
		for key, item := range typed {
			stripped[key] = stripWorkDirValue(item, workDir)
		}

		return stripped
	default:
		return value
	}
}

// decodeJSONValue decodes JSON, keeping numbers exact.
func decodeJSONValue(value json.RawMessage) (any, bool) {
	decoder := json.NewDecoder(bytes.NewReader(value))
	decoder.UseNumber()

	var decoded any
	if err := decoder.Decode(&decoded); err != nil {
		return nil, false
	}

	return decoded, true
}

// stripWorkDirString strips every reference to the working directory.
func stripWorkDirString(value string, workDir string) string {
	output, _ := stripStreamingString(streamingStrip{
		value:          value,
		workDir:        workDir,
		final:          true,
		precedingValue: noPrecedingByte,
	})

	return output
}

// streamingStrip is one step of stripping a streamed string.
type streamingStrip struct {
	value   string
	workDir string

	// final reports that no more text follows. Otherwise a trailing prefix of
	// the working directory is held back as pending.
	final bool

	// precedingValue is the byte before value, or noPrecedingByte.
	precedingValue int
}

// stripStreamingString strips the working directory from value and returns
// the output and the pending suffix that might start a reference.
func stripStreamingString(input streamingStrip) (string, string) {
	remaining := input.value
	before := input.precedingValue

	var output strings.Builder

	for remaining != "" {
		matchIndex := strings.Index(remaining, input.workDir)
		if matchIndex < 0 {
			if input.final {
				output.WriteString(remaining)

				return output.String(), ""
			}

			outputLength := len(remaining) - pendingPrefixLength(remaining, input.workDir, before)
			output.WriteString(remaining[:outputLength])

			return output.String(), remaining[outputLength:]
		}

		followingIndex := matchIndex + len(input.workDir)
		hasBoundary := isPathBoundary(remaining, matchIndex, before)
		atEnd := followingIndex == len(remaining)

		if hasBoundary && atEnd && !input.final {
			output.WriteString(remaining[:matchIndex])

			return output.String(), remaining[matchIndex:]
		}

		followedBySeparator := !atEnd && remaining[followingIndex] == '/'
		isPath := hasBoundary && (atEnd || followedBySeparator || isPathTerminator(remaining[followingIndex]))

		switch {
		case !isPath:
			output.WriteString(remaining[:followingIndex])
			before = int(remaining[followingIndex-1])
			remaining = remaining[followingIndex:]
		case followedBySeparator:
			output.WriteString(remaining[:matchIndex])
			before = '/'
			remaining = remaining[followingIndex+1:]
		default:
			output.WriteString(remaining[:matchIndex])
			output.WriteByte('.')
			remaining = remaining[followingIndex:]
		}
	}

	return output.String(), ""
}

// pendingPrefixLength returns the length of the longest suffix of remaining
// that starts a reference to the working directory at a path boundary.
func pendingPrefixLength(remaining string, workDir string, before int) int {
	length := min(len(remaining), len(workDir)-1)

	for length > 0 {
		start := len(remaining) - length
		startsReference := strings.HasPrefix(workDir, remaining[start:])

		if startsReference && isPathBoundary(remaining, start, before) {
			return length
		}

		length--
	}

	return 0
}

// isPathBoundary reports whether a path may start at index: at the start of
// the text, or after whitespace, a quote or an equals sign.
func isPathBoundary(value string, index int, before int) bool {
	previous := before
	if index > 0 {
		previous = int(value[index-1])
	}

	if previous == noPrecedingByte {
		return true
	}

	return strings.IndexByte(" \t\n\r\f\v\"'`=", byte(previous)) >= 0
}

// isPathTerminator reports whether character ends a path.
func isPathTerminator(character byte) bool {
	return strings.IndexByte(" \t\n\r\f\v\"'`;|&<>()", character) >= 0
}

// toolInputStripper strips the working directory from streamed tool input,
// holding back text that might start a reference across deltas.
type toolInputStripper struct {
	workDir string
	states  map[string]toolInputStripState
}

// toolInputStripState is the pending text of one tool call's input.
type toolInputStripState struct {
	pending string
	before  int
}

// newToolInputStripper returns a stripper for workDir.
func newToolInputStripper(workDir string) *toolInputStripper {
	return &toolInputStripper{
		workDir: workDir,
		states:  make(map[string]toolInputStripState),
	}
}

// strip returns the parts to forward for one tool input part.
func (s *toolInputStripper) strip(part StreamPart) []StreamPart {
	if s.workDir == "" {
		return []StreamPart{
			part,
		}
	}

	switch typed := part.(type) {
	case ToolInputStartPart:
		s.states[typed.ID] = toolInputStripState{
			before: noPrecedingByte,
		}

		return []StreamPart{
			part,
		}
	case ToolInputDeltaPart:
		return s.stripDelta(typed)
	case ToolInputEndPart:
		return s.stripEnd(typed)
	default:
		return []StreamPart{
			part,
		}
	}
}

// stripDelta strips a delta, keeping a possible reference prefix pending.
func (s *toolInputStripper) stripDelta(part ToolInputDeltaPart) []StreamPart {
	state, ok := s.states[part.ID]
	if !ok {
		state = toolInputStripState{
			before: noPrecedingByte,
		}
	}

	value := state.pending + part.Delta
	output, pending := stripStreamingString(streamingStrip{
		value:          value,
		workDir:        s.workDir,
		precedingValue: state.before,
	})

	s.states[part.ID] = toolInputStripState{
		pending: pending,
		before:  byteBefore(value, len(value)-len(pending), state.before),
	}

	if output == "" {
		return nil
	}

	part.Delta = output

	return []StreamPart{
		part,
	}
}

// stripEnd flushes the pending text of a tool call before its end part.
func (s *toolInputStripper) stripEnd(part ToolInputEndPart) []StreamPart {
	state, ok := s.states[part.ID]
	delete(s.states, part.ID)

	if !ok || state.pending == "" {
		return []StreamPart{
			part,
		}
	}

	output, _ := stripStreamingString(streamingStrip{
		value:          state.pending,
		workDir:        s.workDir,
		final:          true,
		precedingValue: state.before,
	})

	if output == "" {
		return []StreamPart{
			part,
		}
	}

	delta := ToolInputDeltaPart{
		ID:    part.ID,
		Delta: output,
	}

	return []StreamPart{
		delta,
		part,
	}
}

// byteBefore returns the byte before index in value, or fallback at the
// start.
func byteBefore(value string, index int, fallback int) int {
	if index <= 0 {
		return fallback
	}

	return int(value[index-1])
}
