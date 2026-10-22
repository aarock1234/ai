package ai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"slices"
	"strconv"

	"github.com/aarock1234/ai/pkg/model"
)

// decisionTolerance is the absolute tolerance of distribution sums and
// weighted means. Provider output is never renormalized.
const decisionTolerance = 1e-6

// maxDecisionRoundingDecimals bounds the declared rounding precision.
const maxDecisionRoundingDecimals = 15

// validateDecisionInput checks the state and the questions of a decision.
func validateDecisionInput(state []model.DecisionStatePart, questions []model.DecisionQuestion) error {
	if err := validateDecisionState(state); err != nil {
		return err
	}

	if len(questions) == 0 {
		return &InvalidArgumentError{
			Parameter: "questions",
			Message:   "must be a nonempty list of questions",
		}
	}

	seen := make(map[string]bool, len(questions))

	for index, question := range questions {
		if question == nil {
			return &InvalidArgumentError{
				Parameter: "questions",
				Value:     index,
				Message:   "question must not be nil",
			}
		}

		if seen[question.QuestionID()] {
			return invalidQuestion(question, fmt.Sprintf("duplicate question ID %q", question.QuestionID()))
		}

		seen[question.QuestionID()] = true

		if err := validateDecisionQuestion(question); err != nil {
			return err
		}
	}

	return nil
}

func validateDecisionState(state []model.DecisionStatePart) error {
	for index, part := range state {
		if isValidDecisionStatePart(part) {
			continue
		}

		return &InvalidArgumentError{
			Parameter: "state",
			Value:     index,
			Message:   "must contain text, file or json parts with valid content",
		}
	}

	return nil
}

func isValidDecisionStatePart(part model.DecisionStatePart) bool {
	switch typed := part.(type) {
	case model.TextPart:
		return true

	case model.FilePart:
		return typed.Data != nil && typed.MediaType != ""

	case model.DecisionJSONPart:
		return json.Valid(typed.Value)

	default:
		return false
	}
}

// validateDecisionQuestion checks the instructions and criteria of one
// question.
func validateDecisionQuestion(question model.DecisionQuestion) error {
	instructions := question.QuestionInstructions()

	if _, isBox := question.(model.BoxQuestion); isBox && instructions == nil {
		return invalidQuestion(question, "box questions require instructions")
	}

	if !isValidDecisionInput(instructions) {
		return invalidQuestion(question, "instructions must be text or a JSON object or array")
	}

	switch typed := question.(type) {
	case model.ChoiceQuestion:
		return validateChoiceQuestion(typed)

	case model.ScoreQuestion:
		return validateScoreQuestion(typed)

	case model.BooleanQuestion:
		return validateBooleanQuestion(typed)

	case model.BoxQuestion:
		return nil

	default:
		return invalidQuestion(question, "question type must be choice, score, boolean, or box")
	}
}

func validateChoiceQuestion(question model.ChoiceQuestion) error {
	if len(question.Choices) == 0 {
		return invalidQuestion(question, "choice criteria must be a nonempty option list")
	}

	names := make(map[string]bool, len(question.Choices))

	for _, choice := range question.Choices {
		if names[choice.Name] {
			return invalidQuestion(question, fmt.Sprintf("duplicate choice %q", choice.Name))
		}

		names[choice.Name] = true

		if !isValidDecisionInput(choice.Description) {
			return invalidCriteria(question)
		}
	}

	return nil
}

func validateScoreQuestion(question model.ScoreQuestion) error {
	if len(question.Levels) == 0 {
		return invalidQuestion(question, "score criteria must contain at least one ordered level")
	}

	for _, level := range question.Levels {
		if !isValidDecisionInput(level) {
			return invalidCriteria(question)
		}
	}

	return nil
}

func validateBooleanQuestion(question model.BooleanQuestion) error {
	if question.Instructions == nil && question.True == nil && question.False == nil {
		return invalidQuestion(question, "boolean questions require instructions or at least one non-null criterion")
	}

	if !isValidDecisionInput(question.True) || !isValidDecisionInput(question.False) {
		return invalidCriteria(question)
	}

	return nil
}

// isValidDecisionInput accepts nil, text, and JSON objects and arrays.
func isValidDecisionInput(input model.DecisionInput) bool {
	switch typed := input.(type) {
	case nil, model.DecisionText:
		return true

	case model.DecisionJSON:
		return isJSONObjectOrArray(typed)

	default:
		return false
	}
}

func isJSONObjectOrArray(value []byte) bool {
	trimmed := bytes.TrimSpace(value)
	if !json.Valid(trimmed) || len(trimmed) == 0 {
		return false
	}

	return trimmed[0] == '{' || trimmed[0] == '['
}

func invalidQuestion(question model.DecisionQuestion, message string) error {
	return &InvalidArgumentError{
		Parameter: "questions." + question.QuestionID(),
		Value:     question,
		Message:   message,
	}
}

func invalidCriteria(question model.DecisionQuestion) error {
	return invalidQuestion(question, "criteria descriptions must be text, JSON objects or arrays, or nil")
}

// decisionAnswerCheck validates the answers of one decision response.
type decisionAnswerCheck struct {
	answers map[string]model.DecisionAnswer

	// probabilityError and scoreError are the rounding errors allowed per
	// rounded value.
	probabilityError float64
	scoreError       float64
}

// validateDecisionAnswers checks that there is exactly one valid answer per
// question, allowing for the declared rounding.
func validateDecisionAnswers(
	questions []model.DecisionQuestion,
	answers map[string]model.DecisionAnswer,
	rounding *model.DecisionRounding,
) error {
	check := decisionAnswerCheck{
		answers: answers,
	}

	if rounding != nil {
		probabilityError, err := roundingError(rounding.ProbabilityDecimals)
		if err != nil {
			return err
		}

		scoreError, err := roundingError(rounding.ScoreDecimals)
		if err != nil {
			return err
		}

		check.probabilityError = probabilityError
		check.scoreError = scoreError
	}

	if !hasAnswerPerQuestion(questions, answers) {
		return invalidDecisionAnswers("decision must return exactly one answer for every question")
	}

	for _, question := range questions {
		if err := check.answer(question); err != nil {
			return err
		}
	}

	return nil
}

func roundingError(decimals *int) (float64, error) {
	if decimals == nil {
		return 0, nil
	}

	if *decimals < 0 || *decimals > maxDecisionRoundingDecimals {
		return 0, invalidDecisionAnswers("decision rounding decimals must be integers between 0 and 15")
	}

	return 0.5 * math.Pow(10, -float64(*decimals)), nil
}

func hasAnswerPerQuestion(questions []model.DecisionQuestion, answers map[string]model.DecisionAnswer) bool {
	if len(answers) != len(questions) {
		return false
	}

	for _, question := range questions {
		if answers[question.QuestionID()] == nil {
			return false
		}
	}

	return true
}

// answer checks the answer to one question. Refusals are valid for any
// question.
func (c decisionAnswerCheck) answer(question model.DecisionQuestion) error {
	id := question.QuestionID()
	answer := c.answers[id]

	if _, isRefusal := answer.(model.RefusalAnswer); isRefusal {
		return nil
	}

	switch typed := question.(type) {
	case model.ChoiceQuestion:
		choice, ok := answer.(model.ChoiceAnswer)
		if !ok {
			return wrongAnswerType(id)
		}

		return c.choice(id, typed, choice)

	case model.ScoreQuestion:
		score, ok := answer.(model.ScoreAnswer)
		if !ok {
			return wrongAnswerType(id)
		}

		return c.score(id, typed, score)

	case model.BoxQuestion:
		box, ok := answer.(model.BoxAnswer)
		if !ok {
			return wrongAnswerType(id)
		}

		return validateBox(id, box.Box)

	case model.BooleanQuestion:
		boolean, ok := answer.(model.BooleanAnswer)
		if !ok {
			return wrongAnswerType(id)
		}

		if !isProbability(boolean.Probability) {
			return invalidDecisionAnswers(fmt.Sprintf("question %q must return P(true) as a finite probability in [0, 1]", id))
		}

		return nil

	default:
		return wrongAnswerType(id)
	}
}

// choice checks that the choice names an option and is the most probable
// one of a complete distribution.
func (c decisionAnswerCheck) choice(id string, question model.ChoiceQuestion, answer model.ChoiceAnswer) error {
	names := make([]string, len(question.Choices))
	for index, choice := range question.Choices {
		names[index] = choice.Name
	}

	if !slices.Contains(names, answer.Choice) {
		return invalidDecisionAnswers(fmt.Sprintf("question %q selected an unknown option", id))
	}

	if answer.Probabilities == nil {
		return nil
	}

	if !hasExactKeys(answer.Probabilities, names) {
		return incompleteDistribution(id)
	}

	if err := c.distribution(id, slices.Collect(maps.Values(answer.Probabilities))); err != nil {
		return err
	}

	selected := answer.Probabilities[answer.Choice]
	for _, probability := range answer.Probabilities {
		if probability > selected+decisionTolerance+2*c.probabilityError {
			return invalidDecisionAnswers(fmt.Sprintf("question %q did not select a highest-probability option", id))
		}
	}

	return nil
}

// score checks that the score lies on the scale and equals the weighted
// mean of its distribution.
func (c decisionAnswerCheck) score(id string, question model.ScoreQuestion, answer model.ScoreAnswer) error {
	lastLevel := float64(len(question.Levels) - 1)

	if !isFinite(answer.Score) || answer.Score < 0 || answer.Score > lastLevel {
		return invalidDecisionAnswers(fmt.Sprintf("question %q score must be in [0, %s]", id, strconv.FormatFloat(lastLevel, 'f', -1, 64)))
	}

	if answer.Probabilities == nil {
		return nil
	}

	if len(answer.Probabilities) != len(question.Levels) {
		return incompleteDistribution(id)
	}

	if err := c.distribution(id, answer.Probabilities); err != nil {
		return err
	}

	mean := 0.0
	meanRoundingError := 0.0

	for level, probability := range answer.Probabilities {
		mean += float64(level) * probability
		meanRoundingError += float64(level) * c.probabilityError
	}

	if math.Abs(mean-answer.Score) > decisionTolerance+meanRoundingError+c.scoreError {
		return invalidDecisionAnswers(fmt.Sprintf(
			"question %q score must equal the probability-weighted mean within the declared rounding precision",
			id,
		))
	}

	return nil
}

// distribution checks that every value is a probability and that they sum
// to one within the rounding precision.
func (c decisionAnswerCheck) distribution(id string, probabilities []float64) error {
	sum := 0.0

	for _, probability := range probabilities {
		if !isProbability(probability) {
			return incompleteDistribution(id)
		}

		sum += probability
	}

	allowed := decisionTolerance + float64(len(probabilities))*c.probabilityError
	if math.Abs(sum-1) > allowed {
		return invalidDecisionAnswers(fmt.Sprintf(
			"question %q probabilities must sum to 1 within the declared rounding precision",
			id,
		))
	}

	return nil
}

func validateBox(id string, box model.DecisionBox) error {
	edges := []float64{
		box.Top,
		box.Bottom,
		box.Left,
		box.Right,
	}

	for _, edge := range edges {
		if !isProbability(edge) {
			return invalidBox(id)
		}
	}

	if box.Left >= box.Right || box.Top >= box.Bottom {
		return invalidBox(id)
	}

	return nil
}

func invalidBox(id string) error {
	return invalidDecisionAnswers(fmt.Sprintf(
		"question %q must return box edges in [0, 1] with left < right and top < bottom",
		id,
	))
}

func hasExactKeys(values map[string]float64, keys []string) bool {
	if len(values) != len(keys) {
		return false
	}

	for _, key := range keys {
		if _, ok := values[key]; !ok {
			return false
		}
	}

	return true
}

func isProbability(value float64) bool {
	return isFinite(value) && value >= 0 && value <= 1
}

func isFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func wrongAnswerType(id string) error {
	return invalidDecisionAnswers(fmt.Sprintf("question %q returned an answer with the wrong type", id))
}

func incompleteDistribution(id string) error {
	return invalidDecisionAnswers(fmt.Sprintf(
		"question %q must have a complete distribution of finite probabilities in [0, 1]",
		id,
	))
}

func invalidDecisionAnswers(message string) error {
	return fmt.Errorf("%w: %s", model.ErrInvalidResponse, message)
}
