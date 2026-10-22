package wirejson

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/aarock1234/ai/internal/decisionwire"
	"github.com/aarock1234/ai/pkg/model"
)

// DecisionJSONPart is an encoded structured data part of a decision state.
type DecisionJSONPart struct {
	Type  string          `json:"type"`
	Value json.RawMessage `json:"value"`
}

func (DecisionJSONPart) isPart() {}

// DecisionState encodes the parts of a decision state, with file contents
// in the encoder's file data encoding.
func (e Encoder) DecisionState(state []model.DecisionStatePart) []Part {
	encoded := make([]Part, 0, len(state))

	for _, statePart := range state {
		if jsonPart, ok := statePart.(model.DecisionJSONPart); ok {
			encoded = append(encoded, DecisionJSONPart{
				Type:  "json",
				Value: RawValue(jsonPart.Value),
			})

			continue
		}

		if part, ok := e.part(statePart); ok {
			encoded = append(encoded, part)
		}
	}

	return encoded
}

// DecisionAnswer is an encoded answer. Which members are set depends on
// the type: "choice", "score", "box", "boolean" or "refusal".
type DecisionAnswer struct {
	Type string `json:"type"`

	Choice      *string           `json:"choice,omitempty"`
	Score       *float64          `json:"score,omitempty"`
	Box         *decisionwire.Box `json:"box,omitempty"`
	Probability *float64          `json:"probability,omitempty"`

	// Probabilities is keyed by choice name, or by level index for scores.
	Probabilities json.RawMessage `json:"probabilities,omitempty"`
}

// namedDecisionAnswer is an answer with the question ID that keys it.
type namedDecisionAnswer struct {
	questionID string
	answer     DecisionAnswer
}

// DecisionAnswers is an ordered set of answers, encoded as a JSON object
// keyed by question ID.
type DecisionAnswers []namedDecisionAnswer

// MarshalJSON encodes the answers as an object in order.
func (a DecisionAnswers) MarshalJSON() ([]byte, error) {
	var buffer bytes.Buffer
	buffer.WriteByte('{')

	for index, named := range a {
		if index > 0 {
			buffer.WriteByte(',')
		}

		key, err := json.Marshal(named.questionID)
		if err != nil {
			return nil, fmt.Errorf("encode decision question id: %w", err)
		}

		value, err := json.Marshal(named.answer)
		if err != nil {
			return nil, fmt.Errorf("encode decision answer %q: %w", named.questionID, err)
		}

		buffer.Write(key)
		buffer.WriteByte(':')
		buffer.Write(value)
	}

	buffer.WriteByte('}')

	return buffer.Bytes(), nil
}

// EncodeDecisionAnswers encodes the answers keyed by question ID, in
// question order. Questions without an answer are left out.
func EncodeDecisionAnswers(
	answers map[string]model.DecisionAnswer,
	questions []model.DecisionQuestion,
) (DecisionAnswers, error) {
	encoded := make(DecisionAnswers, 0, len(answers))

	for _, question := range questions {
		answer, ok := answers[question.QuestionID()]
		if !ok {
			continue
		}

		wireAnswer, err := encodeDecisionAnswer(answer)
		if err != nil {
			return nil, err
		}

		encoded = append(encoded, namedDecisionAnswer{
			questionID: question.QuestionID(),
			answer:     wireAnswer,
		})
	}

	return encoded, nil
}

func encodeDecisionAnswer(answer model.DecisionAnswer) (DecisionAnswer, error) {
	switch typed := answer.(type) {
	case model.ChoiceAnswer:
		return choiceDecisionAnswer(typed)

	case model.ScoreAnswer:
		return scoreDecisionAnswer(typed)

	case model.BoxAnswer:
		box := decisionwire.Box(typed.Box)

		return DecisionAnswer{
			Type: "box",
			Box:  &box,
		}, nil

	case model.BooleanAnswer:
		probability := typed.Probability

		return DecisionAnswer{
			Type:        "boolean",
			Probability: &probability,
		}, nil

	default:
		return DecisionAnswer{
			Type: "refusal",
		}, nil
	}
}

func choiceDecisionAnswer(answer model.ChoiceAnswer) (DecisionAnswer, error) {
	choice := answer.Choice

	encoded := DecisionAnswer{
		Type:   "choice",
		Choice: &choice,
	}

	if answer.Probabilities == nil {
		return encoded, nil
	}

	probabilities, err := json.Marshal(answer.Probabilities)
	if err != nil {
		return DecisionAnswer{}, fmt.Errorf("encode choice probabilities: %w", err)
	}

	encoded.Probabilities = probabilities

	return encoded, nil
}

func scoreDecisionAnswer(answer model.ScoreAnswer) (DecisionAnswer, error) {
	score := answer.Score

	encoded := DecisionAnswer{
		Type:  "score",
		Score: &score,
	}

	if answer.Probabilities == nil {
		return encoded, nil
	}

	probabilities, err := levelProbabilities(answer.Probabilities)
	if err != nil {
		return DecisionAnswer{}, err
	}

	encoded.Probabilities = probabilities

	return encoded, nil
}

// levelProbabilities encodes a score distribution as an object keyed by
// level index, in level order.
func levelProbabilities(probabilities []float64) (json.RawMessage, error) {
	var buffer bytes.Buffer
	buffer.WriteByte('{')

	for level, probability := range probabilities {
		if level > 0 {
			buffer.WriteByte(',')
		}

		value, err := json.Marshal(probability)
		if err != nil {
			return nil, fmt.Errorf("encode level %d probability: %w", level, err)
		}

		buffer.WriteString(strconv.Quote(strconv.Itoa(level)))
		buffer.WriteByte(':')
		buffer.Write(value)
	}

	buffer.WriteByte('}')

	return buffer.Bytes(), nil
}
