package decisionlm

import (
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/model"
)

// decodeAnswers parses the generated JSON object and converts the value of
// every question.
func decodeAnswers(content []model.Content, questions []model.DecisionQuestion) (map[string]model.DecisionAnswer, error) {
	text := generatedText(content)

	var values map[string]json.RawMessage
	if err := json.Unmarshal([]byte(text), &values); err != nil {
		return nil, fmt.Errorf("%w: decision did not return a valid JSON object: %w", model.ErrInvalidResponse, err)
	}

	if values == nil {
		return nil, fmt.Errorf("%w: decision did not return a valid JSON object", model.ErrInvalidResponse)
	}

	if len(values) != len(questions) {
		return nil, exactlyOneValueError()
	}

	answers := make(map[string]model.DecisionAnswer, len(questions))

	for index, question := range questions {
		value, ok := values[questionKey(index)]
		if !ok {
			return nil, exactlyOneValueError()
		}

		answer, err := convertAnswer(question, value)
		if err != nil {
			return nil, err
		}

		answers[question.QuestionID()] = answer
	}

	return answers, nil
}

// generatedText joins the text content.
func generatedText(content []model.Content) string {
	var text strings.Builder

	for _, part := range content {
		if typed, ok := part.(model.Text); ok {
			text.WriteString(typed.Text)
		}
	}

	return text.String()
}

func convertAnswer(question model.DecisionQuestion, value json.RawMessage) (model.DecisionAnswer, error) {
	switch typed := question.(type) {
	case model.ChoiceQuestion:
		return choiceAnswer(typed, value)

	case model.BooleanQuestion:
		probability, ok := finiteNumber(value)
		isProbability := ok && probability >= 0 && probability <= 1

		if !isProbability {
			return nil, fmt.Errorf(
				"%w: question %q must return P(true) as a finite probability in [0, 1]",
				model.ErrInvalidResponse,
				typed.ID,
			)
		}

		return model.BooleanAnswer{
			Probability: probability,
		}, nil

	case model.ScoreQuestion:
		score, ok := finiteNumber(value)
		highestLevel := float64(len(typed.Levels) - 1)
		isOnScale := ok && score >= 0 && score <= highestLevel

		if !isOnScale {
			return nil, fmt.Errorf("%w: question %q returned a score outside its rubric", model.ErrInvalidResponse, typed.ID)
		}

		return model.ScoreAnswer{
			Score: score,
		}, nil

	default:
		return nil, fmt.Errorf("%w: question %q has an unsupported type", model.ErrInvalidResponse, question.QuestionID())
	}
}

// choiceAnswer accepts an internal option code or, as a fallback, the
// option's label.
func choiceAnswer(question model.ChoiceQuestion, value json.RawMessage) (model.DecisionAnswer, error) {
	var selected string
	if wirejson.IsNull(value) || json.Unmarshal(value, &selected) != nil {
		return nil, unknownOptionError(question.ID)
	}

	index := choiceCodeIndex(selected, len(question.Choices))

	if index == -1 {
		index = slices.IndexFunc(question.Choices, func(choice model.DecisionChoice) bool {
			return choice.Name == selected
		})
	}

	if index == -1 {
		return nil, unknownOptionError(question.ID)
	}

	return model.ChoiceAnswer{
		Choice: question.Choices[index].Name,
	}, nil
}

// choiceCodeIndex returns the index of the option with the internal code,
// or -1.
func choiceCodeIndex(code string, count int) int {
	for index := range count {
		if code == choiceKey(index) {
			return index
		}
	}

	return -1
}

// finiteNumber decodes a finite JSON number.
func finiteNumber(value json.RawMessage) (float64, bool) {
	var number float64
	if wirejson.IsNull(value) || json.Unmarshal(value, &number) != nil {
		return 0, false
	}

	return number, !math.IsNaN(number) && !math.IsInf(number, 0)
}

func exactlyOneValueError() error {
	return fmt.Errorf("%w: decision must return exactly one value per question", model.ErrInvalidResponse)
}

func unknownOptionError(questionID string) error {
	return fmt.Errorf("%w: question %q selected an unknown option", model.ErrInvalidResponse, questionID)
}
