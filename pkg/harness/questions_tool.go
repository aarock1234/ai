package harness

import (
	"encoding/json"
	"fmt"
)

// questionsToolDescription describes the common askUserQuestions tool.
const questionsToolDescription = "Ask the user one or more questions"

// QuestionsToolInputSchema is the JSON Schema of QuestionsToolInput, the
// input of the common askUserQuestions tool.
const QuestionsToolInputSchema = `{
  "type": "object",
  "properties": {
    "allowPartialAnswers": {"type": "boolean"},
    "questions": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "string", "minLength": 1},
          "question": {"type": "string", "minLength": 1},
          "header": {"type": "string"},
          "options": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "id": {"type": "string", "minLength": 1},
                "label": {"type": "string", "minLength": 1},
                "description": {"type": "string"},
                "preview": {"type": "string"}
              },
              "required": ["id", "label"],
              "additionalProperties": false
            }
          },
          "allowMultiple": {"type": "boolean"},
          "allowFreeForm": {
            "anyOf": [
              {"type": "boolean"},
              {
                "type": "object",
                "properties": {"secret": {"type": "boolean"}},
                "required": ["secret"],
                "additionalProperties": false
              }
            ]
          }
        },
        "required": ["id", "question"],
        "additionalProperties": false
      }
    }
  },
  "required": ["allowPartialAnswers", "questions"],
  "additionalProperties": false
}`

// QuestionsToolOutputSchema is the JSON Schema of QuestionsToolOutput.
const QuestionsToolOutputSchema = `{
  "oneOf": [
    {
      "type": "object",
      "properties": {
        "action": {"const": "answered"},
        "answers": {"type": "object", "additionalProperties": {"$ref": "#/$defs/answer"}}
      },
      "required": ["action", "answers"]
    },
    {
      "type": "object",
      "properties": {
        "action": {"const": "partially-answered"},
        "answers": {"type": "object", "additionalProperties": {"$ref": "#/$defs/answer"}}
      },
      "required": ["action", "answers"]
    },
    {
      "type": "object",
      "properties": {"action": {"const": "declined"}},
      "required": ["action"]
    },
    {
      "type": "object",
      "properties": {"action": {"const": "cancelled"}},
      "required": ["action"]
    }
  ],
  "$defs": {
    "answer": {
      "type": "object",
      "properties": {
        "optionIds": {"type": "array", "items": {"type": "string"}},
        "freeform": {"type": "string"}
      },
      "required": ["optionIds"]
    }
  }
}`

// QuestionsToolInput is the input of the common askUserQuestions tool. The
// agent never executes the tool: it pauses the turn so that the application
// can answer through a tool result.
type QuestionsToolInput struct {
	AllowPartialAnswers bool       `json:"allowPartialAnswers"`
	Questions           []Question `json:"questions"`
}

// Question is one question of QuestionsToolInput.
type Question struct {
	ID       string           `json:"id"`
	Question string           `json:"question"`
	Header   string           `json:"header,omitempty"`
	Options  []QuestionOption `json:"options,omitempty"`

	AllowMultiple bool `json:"allowMultiple,omitempty"`

	AllowFreeForm QuestionFreeForm `json:"allowFreeForm,omitzero"`
}

// QuestionOption is one choice of a Question.
type QuestionOption struct {
	ID          string `json:"id"`
	Label       string `json:"label"`
	Description string `json:"description,omitempty"`
	Preview     string `json:"preview,omitempty"`
}

// QuestionFreeForm controls free-form answers. It encodes as a boolean, or
// as an object with the secret flag for secret answers.
type QuestionFreeForm struct {
	// Allowed allows a free-form answer.
	Allowed bool

	// Secret asks the client to hide the answer as it is typed. It implies
	// Allowed.
	Secret bool
}

// questionFreeFormJSON is the object form of QuestionFreeForm.
type questionFreeFormJSON struct {
	Secret bool `json:"secret"`
}

// MarshalJSON encodes a boolean, or the object form for secret answers.
func (f QuestionFreeForm) MarshalJSON() ([]byte, error) {
	if f.Secret {
		return json.Marshal(questionFreeFormJSON{
			Secret: true,
		})
	}

	return json.Marshal(f.Allowed)
}

// UnmarshalJSON decodes a boolean or the object form.
func (f *QuestionFreeForm) UnmarshalJSON(data []byte) error {
	var allowed bool
	if err := json.Unmarshal(data, &allowed); err == nil {
		*f = QuestionFreeForm{
			Allowed: allowed,
		}

		return nil
	}

	var object questionFreeFormJSON
	if err := json.Unmarshal(data, &object); err != nil {
		return fmt.Errorf("decode allowFreeForm: %w", err)
	}

	*f = QuestionFreeForm{
		Allowed: true,
		Secret:  object.Secret,
	}

	return nil
}

// QuestionsAction is how the user handled the questions.
type QuestionsAction string

// Questions actions.
const (
	QuestionsAnswered          QuestionsAction = "answered"
	QuestionsPartiallyAnswered QuestionsAction = "partially-answered"
	QuestionsDeclined          QuestionsAction = "declined"
	QuestionsCancelled         QuestionsAction = "cancelled"
)

// QuestionsToolOutput is the result of the askUserQuestions tool.
type QuestionsToolOutput struct {
	Action QuestionsAction `json:"action"`

	// Answers maps question IDs to answers. It is set for the answered and
	// partially answered actions.
	Answers map[string]QuestionAnswer `json:"answers,omitempty"`
}

// QuestionAnswer is the answer to one question.
type QuestionAnswer struct {
	OptionIDs []string `json:"optionIds"`
	Freeform  string   `json:"freeform,omitempty"`
}
