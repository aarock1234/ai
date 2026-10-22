package model

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// DecisionModelKind names decision models in errors and registries.
const DecisionModelKind Kind = "decision model"

// DecisionModel answers typed questions about one shared state, such as
// choosing an option, scoring on a scale or estimating the probability of a
// statement. The decision contract is experimental and may change.
type DecisionModel interface {
	Provider() string
	ModelID() string

	// SupportedQuestionTypes lists the question types the model answers, so
	// that unsupported calls are rejected before any request is sent.
	SupportedQuestionTypes() []DecisionQuestionType

	// Decide answers every question against the same state. It returns no
	// partial results.
	Decide(ctx context.Context, options DecisionOptions) (*DecisionResult, error)
}

// DecisionProvider is implemented by providers that offer decision models.
// It is separate from Provider because most providers have none.
type DecisionProvider interface {
	DecisionModel(modelID string) (DecisionModel, error)
}

// DecisionOptions are the inputs of one decision call.
type DecisionOptions struct {
	State     []DecisionStatePart
	Questions []DecisionQuestion

	// MaxOutputTokens limits generated tokens on models that support it.
	MaxOutputTokens *int

	// Reasoning selects a reasoning effort on models that support it.
	Reasoning ReasoningEffort

	Headers         http.Header
	ProviderOptions ProviderOptions
}

// DecisionStatePart is one part of the shared state: TextPart, FilePart or
// DecisionJSONPart.
type DecisionStatePart interface {
	isDecisionStatePart()
}

// DecisionJSONPart is structured data in the shared state.
type DecisionJSONPart struct {
	Value json.RawMessage
}

func (TextPart) isDecisionStatePart()         {}
func (FilePart) isDecisionStatePart()         {}
func (DecisionJSONPart) isDecisionStatePart() {}

// DecisionInput is the content of instructions and criteria descriptions:
// DecisionText or DecisionJSON. A nil DecisionInput means no description.
type DecisionInput interface {
	isDecisionInput()
}

// DecisionText is plain text.
type DecisionText string

// DecisionJSON is a JSON object or array.
type DecisionJSON json.RawMessage

func (DecisionText) isDecisionInput() {}
func (DecisionJSON) isDecisionInput() {}

// DecisionInputText returns input as text: DecisionText as is and
// DecisionJSON as compact JSON, or unchanged when it is not valid JSON. It
// returns an empty string for nil.
func DecisionInputText(input DecisionInput) string {
	switch typed := input.(type) {
	case DecisionText:
		return string(typed)

	case DecisionJSON:
		var compact bytes.Buffer
		if err := json.Compact(&compact, typed); err != nil {
			return string(typed)
		}

		return compact.String()

	default:
		return ""
	}
}

// DecisionQuestionType is the kind of judgment a question asks for.
type DecisionQuestionType string

// Decision question types.
const (
	DecisionQuestionChoice  DecisionQuestionType = "choice"
	DecisionQuestionScore   DecisionQuestionType = "score"
	DecisionQuestionBox     DecisionQuestionType = "box"
	DecisionQuestionBoolean DecisionQuestionType = "boolean"
)

// DecisionQuestion is a judgment to make about the shared state:
// ChoiceQuestion, ScoreQuestion, BoxQuestion or BooleanQuestion.
type DecisionQuestion interface {
	// QuestionID returns the ID that keys the question's answer.
	QuestionID() string

	// QuestionType returns the kind of judgment.
	QuestionType() DecisionQuestionType

	// QuestionInstructions returns the instructions, or nil.
	QuestionInstructions() DecisionInput
}

// ChoiceQuestion selects one of several named options.
type ChoiceQuestion struct {
	ID string

	// Instructions may be nil when the choices define the judgment and the
	// provider supports it.
	Instructions DecisionInput

	// Choices are the options, in order. Their names are unique.
	Choices []DecisionChoice
}

// DecisionChoice is one option of a ChoiceQuestion.
type DecisionChoice struct {
	Name string

	// Description is nil when the option has none.
	Description DecisionInput
}

// ScoreQuestion places the state on an ordered scale of levels, indexed
// from zero.
type ScoreQuestion struct {
	ID string

	// Instructions may be nil when the levels define the judgment and the
	// provider supports it.
	Instructions DecisionInput

	// Levels describe each level in order. A nil level has no description.
	// Providers may require more than one level.
	Levels []DecisionInput
}

// BoxQuestion locates a region of an image in the state.
type BoxQuestion struct {
	ID           string
	Instructions DecisionInput
}

// BooleanQuestion estimates the probability that a statement is true.
type BooleanQuestion struct {
	ID string

	// Instructions may be nil when a criterion is set.
	Instructions DecisionInput

	// True and False optionally describe each outcome.
	True  DecisionInput
	False DecisionInput
}

// QuestionID returns the question ID.
func (q ChoiceQuestion) QuestionID() string {
	return q.ID
}

// QuestionType returns DecisionQuestionChoice.
func (ChoiceQuestion) QuestionType() DecisionQuestionType {
	return DecisionQuestionChoice
}

// QuestionInstructions returns the instructions.
func (q ChoiceQuestion) QuestionInstructions() DecisionInput {
	return q.Instructions
}

// QuestionID returns the question ID.
func (q ScoreQuestion) QuestionID() string {
	return q.ID
}

// QuestionType returns DecisionQuestionScore.
func (ScoreQuestion) QuestionType() DecisionQuestionType {
	return DecisionQuestionScore
}

// QuestionInstructions returns the instructions.
func (q ScoreQuestion) QuestionInstructions() DecisionInput {
	return q.Instructions
}

// QuestionID returns the question ID.
func (q BoxQuestion) QuestionID() string {
	return q.ID
}

// QuestionType returns DecisionQuestionBox.
func (BoxQuestion) QuestionType() DecisionQuestionType {
	return DecisionQuestionBox
}

// QuestionInstructions returns the instructions.
func (q BoxQuestion) QuestionInstructions() DecisionInput {
	return q.Instructions
}

// QuestionID returns the question ID.
func (q BooleanQuestion) QuestionID() string {
	return q.ID
}

// QuestionType returns DecisionQuestionBoolean.
func (BooleanQuestion) QuestionType() DecisionQuestionType {
	return DecisionQuestionBoolean
}

// QuestionInstructions returns the instructions.
func (q BooleanQuestion) QuestionInstructions() DecisionInput {
	return q.Instructions
}

// DecisionAnswer is the answer to one question: ChoiceAnswer, ScoreAnswer,
// BoxAnswer, BooleanAnswer or RefusalAnswer.
type DecisionAnswer interface {
	isDecisionAnswer()
}

// ChoiceAnswer is the selected option of a ChoiceQuestion.
type ChoiceAnswer struct {
	// Choice is the selected option, the most probable one when a
	// distribution is reported.
	Choice string

	// Probabilities is the complete distribution over the options, or nil
	// when it is not reported.
	Probabilities map[string]float64
}

// ScoreAnswer is the position on the scale of a ScoreQuestion.
type ScoreAnswer struct {
	// Score is a fractional position between zero and the last level.
	Score float64

	// Probabilities is the complete distribution over the levels, indexed
	// by level, or nil when it is not reported. Score is its
	// probability-weighted mean.
	Probabilities []float64
}

// BoxAnswer is the region located by a BoxQuestion.
type BoxAnswer struct {
	Box DecisionBox
}

// DecisionBox is a region of an image. Edges are between 0 and 1, measured
// from the top-left corner.
type DecisionBox struct {
	Top    float64
	Bottom float64
	Left   float64
	Right  float64
}

// BooleanAnswer is the probability that the statement of a BooleanQuestion
// is true. It is not a confidence in either outcome.
type BooleanAnswer struct {
	Probability float64
}

// RefusalAnswer reports that the model declined to answer. It is valid for
// any question type.
type RefusalAnswer struct{}

func (ChoiceAnswer) isDecisionAnswer()  {}
func (ScoreAnswer) isDecisionAnswer()   {}
func (BoxAnswer) isDecisionAnswer()     {}
func (BooleanAnswer) isDecisionAnswer() {}
func (RefusalAnswer) isDecisionAnswer() {}

// DecisionResult is the outcome of one decision call.
type DecisionResult struct {
	// Answers holds exactly one answer per question, keyed by question ID.
	Answers map[string]DecisionAnswer

	// Rounding is nil when values are reported at full precision.
	Rounding *DecisionRounding

	// Usage is nil when the provider did not report it.
	Usage *DecisionUsage

	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Response         ResponseInfo
}

// DecisionRounding declares the decimal places a provider rounds to. The
// caller allows half a unit in the last place per rounded value when it
// checks distribution sums and weighted scores.
type DecisionRounding struct {
	ProbabilityDecimals *int
	ScoreDecimals       *int
}

// DecisionUsage reports token consumption. Nil counts were not reported.
type DecisionUsage struct {
	InputTokens  *int
	OutputTokens *int
}

// DecisionRefusalError reports that a decision model declined to answer one
// or more questions.
type DecisionRefusalError struct {
	QuestionIDs []string
	Provider    string
	ModelID     string
}

// Error names the model and the refused questions.
func (e *DecisionRefusalError) Error() string {
	noun := "questions"
	if len(e.QuestionIDs) == 1 {
		noun = "question"
	}

	quoted := make([]string, len(e.QuestionIDs))
	for index, questionID := range e.QuestionIDs {
		quoted[index] = fmt.Sprintf("%q", questionID)
	}

	return fmt.Sprintf(
		"decision model %q from provider %q refused %s %s",
		e.ModelID,
		e.Provider,
		noun,
		strings.Join(quoted, ", "),
	)
}

// DecisionUnsupportedQuestionTypeError reports a question whose type the
// decision model does not support. It wraps ErrUnsupported.
type DecisionUnsupportedQuestionTypeError struct {
	QuestionID   string
	QuestionType DecisionQuestionType
	Provider     string
	ModelID      string
}

// Error names the question, its type and the model.
func (e *DecisionUnsupportedQuestionTypeError) Error() string {
	return fmt.Sprintf(
		"question %q has type %q, which is not supported by provider %q and model %q",
		e.QuestionID,
		e.QuestionType,
		e.Provider,
		e.ModelID,
	)
}

// Unwrap returns ErrUnsupported.
func (e *DecisionUnsupportedQuestionTypeError) Unwrap() error {
	return ErrUnsupported
}
