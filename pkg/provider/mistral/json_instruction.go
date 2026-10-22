package mistral

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

// schemaPlaceholder is replaced with the JSON schema in the schema
// instruction.
const schemaPlaceholder = "{{schema}}"

var (
	//go:embed prompts/json_instruction.md
	jsonInstruction string

	//go:embed prompts/json_schema_instruction.md
	jsonSchemaInstruction string
)

// injectJSONInstruction adds an instruction to answer with JSON, and the
// schema when there is one, to the system message. It adds a system message
// when the prompt does not start with one. The caller's prompt is not
// modified.
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
	if schema == nil {
		return strings.TrimSpace(jsonInstruction), nil
	}

	encoded, err := json.Marshal(schema)
	if err != nil {
		return "", fmt.Errorf("%w: invalid response format schema: %w", model.ErrInvalidArgument, err)
	}

	instruction := strings.TrimSpace(jsonSchemaInstruction)

	return strings.Replace(instruction, schemaPlaceholder, string(encoded), 1), nil
}
