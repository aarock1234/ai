package openai

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// decisionRequestBody is the body of a Decisions API request.
type decisionRequestBody struct {
	Model            string                 `json:"model"`
	SafetyIdentifier string                 `json:"safety_identifier,omitempty"`
	Input            []decisionInputMessage `json:"input"`
	Questions        []decisionQuestion     `json:"questions"`
}

// decisionInputMessage holds the shared state.
type decisionInputMessage struct {
	Role    string            `json:"role"`
	Content []decisionContent `json:"content"`
}

// decisionContent is an input_text or input_image part.
type decisionContent struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
	Detail   string `json:"detail,omitempty"`
}

// decisionQuestion is a predicate, choice or score question.
type decisionQuestion struct {
	Type         string           `json:"type"`
	Name         string           `json:"name"`
	Instructions *string          `json:"instructions,omitempty"`
	Choices      []decisionChoice `json:"choices,omitempty"`
	Levels       []decisionLevel  `json:"levels,omitempty"`
}

// decisionChoice is one option of a choice question.
type decisionChoice struct {
	Value       string `json:"value"`
	Description string `json:"description,omitempty"`
}

// decisionLevel is one level of a score question.
type decisionLevel struct {
	Label       string `json:"label"`
	Description string `json:"description,omitempty"`
}

// decisionResponse is the body of a Decisions API response.
type decisionResponse struct {
	Model   string           `json:"model"`
	Usage   *decisionUsage   `json:"usage"`
	Answers []decisionAnswer `json:"answers"`
}

// decisionUsage is the token usage of a decision. Raw keeps the original
// object, including the details not modeled here.
type decisionUsage struct {
	InputTokens  *int `json:"input_tokens"`
	OutputTokens *int `json:"output_tokens"`

	Raw json.RawMessage `json:"-"`
}

// UnmarshalJSON decodes the usage and keeps the raw object.
func (u *decisionUsage) UnmarshalJSON(data []byte) error {
	type plain decisionUsage

	var decoded plain
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}

	*u = decisionUsage(decoded)
	u.Raw = append(json.RawMessage(nil), data...)

	return nil
}

// rawUsage returns the API's usage object, or nil.
func (u *decisionUsage) rawUsage() json.RawMessage {
	if u == nil {
		return nil
	}

	return u.Raw
}

// decisionUsage converts the usage. Nil usage was not reported.
func (u *decisionUsage) decisionUsage() *model.DecisionUsage {
	if u == nil {
		return nil
	}

	return &model.DecisionUsage{
		InputTokens:  u.InputTokens,
		OutputTokens: u.OutputTokens,
	}
}

// decisionAnswer is one answer, decoded by its type: a refusal, a predicate
// probability, a choice or a score.
type decisionAnswer struct {
	Type string
	Name string

	// RefusalName is the name of a refused question, which may be null.
	RefusalName *string

	Probability *float64

	Choice              string
	ChoiceProbabilities []decisionChoiceProbability

	Score              float64
	ScoreProbabilities []decisionScoreProbability

	Confidence *float64
}

type decisionChoiceProbability struct {
	Value       string  `json:"value"`
	Probability float64 `json:"probability"`
}

type decisionScoreProbability struct {
	Value       int     `json:"value"`
	Probability float64 `json:"probability"`
}

type decisionRefusalWire struct {
	Name *string `json:"name"`
}

type decisionPredicateWire struct {
	Name        string   `json:"name"`
	Probability *float64 `json:"probability"`
}

type decisionChoiceWire struct {
	Name          string                      `json:"name"`
	Choice        string                      `json:"choice"`
	Confidence    *float64                    `json:"confidence"`
	Probabilities []decisionChoiceProbability `json:"probabilities"`
}

type decisionScoreWire struct {
	Name          string                     `json:"name"`
	Score         float64                    `json:"score"`
	Confidence    *float64                   `json:"confidence"`
	Probabilities []decisionScoreProbability `json:"probabilities"`
}

// UnmarshalJSON decodes an answer by its type.
func (a *decisionAnswer) UnmarshalJSON(data []byte) error {
	var header struct {
		Type string `json:"type"`
	}

	if err := json.Unmarshal(data, &header); err != nil {
		return err
	}

	a.Type = header.Type

	switch header.Type {
	case "refusal":
		var wire decisionRefusalWire
		if err := json.Unmarshal(data, &wire); err != nil {
			return err
		}

		a.RefusalName = wire.Name
		if wire.Name != nil {
			a.Name = *wire.Name
		}

	case "predicate":
		var wire decisionPredicateWire
		if err := json.Unmarshal(data, &wire); err != nil {
			return err
		}

		a.Name = wire.Name
		a.Probability = wire.Probability

	case "choice":
		var wire decisionChoiceWire
		if err := json.Unmarshal(data, &wire); err != nil {
			return err
		}

		a.Name = wire.Name
		a.Choice = wire.Choice
		a.Confidence = wire.Confidence
		a.ChoiceProbabilities = wire.Probabilities

	case "score":
		var wire decisionScoreWire
		if err := json.Unmarshal(data, &wire); err != nil {
			return err
		}

		a.Name = wire.Name
		a.Score = wire.Score
		a.Confidence = wire.Confidence
		a.ScoreProbabilities = wire.Probabilities

	default:
		return fmt.Errorf("unknown decision answer type %q", header.Type)
	}

	return nil
}
