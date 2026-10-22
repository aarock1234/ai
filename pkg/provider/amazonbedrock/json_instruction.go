package amazonbedrock

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/aarock1234/ai/pkg/model"
)

// schemaPlaceholder is replaced with the JSON schema in the instruction.
const schemaPlaceholder = "{{schema}}"

//go:embed prompts/json_instruction.md
var jsonInstruction string

// injectJSONInstruction appends an instruction to answer with only a JSON
// object matching schema to the first system message, adding a system
// message when the prompt does not start with one. The caller's prompt is
// not modified.
func injectJSONInstruction(prompt []model.Message, schema json.RawMessage) ([]model.Message, error) {
	instruction, err := jsonInstructionText(schema)
	if err != nil {
		return nil, err
	}

	system := model.SystemMessage{}
	rest := prompt

	if len(prompt) > 0 {
		if first, ok := prompt[0].(model.SystemMessage); ok {
			system = first
			rest = prompt[1:]
		}
	}

	if system.Content == "" {
		system.Content = instruction
	} else {
		system.Content = system.Content + "\n\n" + instruction
	}

	injected := make([]model.Message, 0, len(rest)+1)
	injected = append(injected, system)

	return append(injected, rest...), nil
}

func jsonInstructionText(schema json.RawMessage) (string, error) {
	encoded, err := json.Marshal(schema)
	if err != nil {
		return "", fmt.Errorf("%w: invalid response format schema: %w", model.ErrInvalidArgument, err)
	}

	instruction := strings.TrimSpace(jsonInstruction)

	return strings.Replace(instruction, schemaPlaceholder, string(encoded), 1), nil
}

// jsonObjectExtractor passes through the text of the first JSON object in
// a text stream, dropping text before and after it, such as markdown
// fences.
type jsonObjectExtractor struct {
	started   bool
	completed bool
	depth     int
	inString  bool
	escaped   bool
}

// process returns the part of text that belongs to the JSON object.
func (e *jsonObjectExtractor) process(text string) string {
	var result strings.Builder

	for len(text) > 0 && !e.completed {
		character, size := utf8.DecodeRuneInString(text)
		piece := text[:size]
		text = text[size:]

		if !e.started {
			if character != '{' {
				continue
			}

			e.started = true
			e.depth = 1
			result.WriteString(piece)

			continue
		}

		result.WriteString(piece)
		e.track(character)
	}

	return result.String()
}

// track updates the nesting state after a character of the object.
func (e *jsonObjectExtractor) track(character rune) {
	if e.escaped {
		e.escaped = false

		return
	}

	if character == '\\' && e.inString {
		e.escaped = true

		return
	}

	if character == '"' {
		e.inString = !e.inString

		return
	}

	if e.inString {
		return
	}

	switch character {
	case '{':
		e.depth++

	case '}':
		e.depth--
		if e.depth == 0 {
			e.completed = true
		}
	}
}
