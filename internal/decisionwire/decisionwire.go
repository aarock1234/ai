// Package decisionwire encodes decision questions and decodes decision
// answers in the record-based JSON form that the AI Gateway and several
// decision APIs share: questions keyed by ID with "instructions" and
// "criteria" members, and answers keyed by question ID.
package decisionwire

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// Input encodes a decision input: text as a JSON string, JSON unchanged,
// and nil as JSON null.
func Input(input model.DecisionInput) json.RawMessage {
	switch typed := input.(type) {
	case model.DecisionText:
		encoded, err := json.Marshal(string(typed))
		if err != nil {
			return json.RawMessage("null")
		}

		return encoded

	case model.DecisionJSON:
		return json.RawMessage(typed)

	default:
		return json.RawMessage("null")
	}
}

// optionalInput encodes an input, or returns nil for a nil input so that
// the member is omitted.
func optionalInput(input model.DecisionInput) json.RawMessage {
	if input == nil {
		return nil
	}

	return Input(input)
}

// Question is a question on the wire. Type is the provider's name of the
// question type.
type Question struct {
	Type         string          `json:"type"`
	Instructions json.RawMessage `json:"instructions,omitempty"`
	Criteria     json.RawMessage `json:"criteria,omitempty"`
}

// NamedQuestion is a question with the ID that keys it.
type NamedQuestion struct {
	ID       string
	Question Question
}

// Questions is an ordered set of questions, encoded as a JSON object keyed
// by question ID in question order.
type Questions []NamedQuestion

// MarshalJSON encodes the questions as an object in order.
func (q Questions) MarshalJSON() ([]byte, error) {
	object := make(jsonobject.Object, 0, len(q))

	for _, named := range q {
		value, err := json.Marshal(named.Question)
		if err != nil {
			return nil, fmt.Errorf("encode question %q: %w", named.ID, err)
		}

		object = append(object, jsonobject.Member{
			Key:   named.ID,
			Value: value,
		})
	}

	return object.Encode()
}

// QuestionTypeNames names the question types on the wire. A missing type
// is unsupported.
type QuestionTypeNames map[model.DecisionQuestionType]string

// NormalizedQuestionTypes are the AI SDK's own names.
var NormalizedQuestionTypes = QuestionTypeNames{
	model.DecisionQuestionChoice:  "choice",
	model.DecisionQuestionScore:   "score",
	model.DecisionQuestionBox:     "box",
	model.DecisionQuestionBoolean: "boolean",
}

// EncodeQuestion converts a question. Choice criteria are an object keyed
// by choice name, score criteria an array of levels, and boolean criteria an
// object with optional "true" and "false" members. It reports false when
// names has no name for the question's type.
func EncodeQuestion(question model.DecisionQuestion, names QuestionTypeNames) (Question, bool, error) {
	typeName, ok := names[question.QuestionType()]
	if !ok {
		return Question{}, false, nil
	}

	criteria, err := encodeCriteria(question)
	if err != nil {
		return Question{}, false, err
	}

	return Question{
		Type:         typeName,
		Instructions: optionalInput(question.QuestionInstructions()),
		Criteria:     criteria,
	}, true, nil
}

// EncodeQuestions encodes questions with the AI SDK's own type names, as the
// provider specification's object keyed by question ID in question order.
// A question of a type without a name is unsupported.
func EncodeQuestions(questions []model.DecisionQuestion) (Questions, error) {
	encoded := make(Questions, 0, len(questions))

	for _, question := range questions {
		wireQuestion, supported, err := EncodeQuestion(question, NormalizedQuestionTypes)
		if err != nil {
			return nil, err
		}

		if !supported {
			return nil, fmt.Errorf(
				"%w: question %q has type %q",
				model.ErrUnsupported,
				question.QuestionID(),
				question.QuestionType(),
			)
		}

		encoded = append(encoded, NamedQuestion{
			ID:       question.QuestionID(),
			Question: wireQuestion,
		})
	}

	return encoded, nil
}

func encodeCriteria(question model.DecisionQuestion) (json.RawMessage, error) {
	switch typed := question.(type) {
	case model.ChoiceQuestion:
		return choiceCriteria(typed.Choices)

	case model.ScoreQuestion:
		levels := make([]json.RawMessage, len(typed.Levels))
		for index, level := range typed.Levels {
			levels[index] = Input(level)
		}

		return json.Marshal(levels)

	case model.BooleanQuestion:
		return booleanCriteria(typed)

	default:
		return nil, nil
	}
}

// choiceCriteria encodes the choices as an object in choice order.
func choiceCriteria(choices []model.DecisionChoice) (json.RawMessage, error) {
	object := make(jsonobject.Object, 0, len(choices))

	for _, choice := range choices {
		object = append(object, jsonobject.Member{
			Key:   choice.Name,
			Value: Input(choice.Description),
		})
	}

	return object.Encode()
}

// booleanOutcomes are the criteria of a boolean question.
type booleanOutcomes struct {
	True  json.RawMessage `json:"true,omitempty"`
	False json.RawMessage `json:"false,omitempty"`
}

func booleanCriteria(question model.BooleanQuestion) (json.RawMessage, error) {
	if question.True == nil && question.False == nil {
		return nil, nil
	}

	return json.Marshal(booleanOutcomes{
		True:  optionalInput(question.True),
		False: optionalInput(question.False),
	})
}

// Box is a region of an image on the wire.
type Box struct {
	Top    float64 `json:"top"`
	Bottom float64 `json:"bottom"`
	Left   float64 `json:"left"`
	Right  float64 `json:"right"`
}

// Answer is an answer on the wire. Which members are set depends on the
// type: "choice", "score", "box", "boolean", "noul" or "refusal".
type Answer struct {
	Type string `json:"type"`

	Choice      *string  `json:"choice"`
	Score       *float64 `json:"score"`
	Probability *float64 `json:"probability"`
	Noul        *float64 `json:"noul"`
	Box         *Box     `json:"box"`

	// Probabilities is keyed by choice name, or by level index for scores.
	Probabilities map[string]float64 `json:"probabilities"`

	Confidence *float64 `json:"confidence"`
}

// Convert returns the normalized answer to question. Boolean answers carry
// their probability in "probability" or, for "noul" answers, in "noul".
func (a Answer) Convert(question model.DecisionQuestion) (model.DecisionAnswer, error) {
	switch a.Type {
	case "choice":
		if a.Choice == nil {
			return nil, invalidAnswer("choice answer has no choice")
		}

		return model.ChoiceAnswer{
			Choice:        *a.Choice,
			Probabilities: a.Probabilities,
		}, nil

	case "score":
		return a.scoreAnswer(ScoreLevelCount(question))

	case "box":
		if a.Box == nil {
			return nil, invalidAnswer("box answer has no box")
		}

		return model.BoxAnswer{
			Box: model.DecisionBox(*a.Box),
		}, nil

	case "boolean":
		return booleanAnswer(a.Probability)

	case "noul":
		return booleanAnswer(a.Noul)

	case "refusal":
		return model.RefusalAnswer{}, nil

	default:
		return nil, invalidAnswer(fmt.Sprintf("unknown answer type %q", a.Type))
	}
}

func (a Answer) scoreAnswer(levelCount int) (model.DecisionAnswer, error) {
	if a.Score == nil {
		return nil, invalidAnswer("score answer has no score")
	}

	probabilities, err := ScoreProbabilities(a.Probabilities, levelCount)
	if err != nil {
		return nil, err
	}

	return model.ScoreAnswer{
		Score:         *a.Score,
		Probabilities: probabilities,
	}, nil
}

func booleanAnswer(probability *float64) (model.DecisionAnswer, error) {
	if probability == nil {
		return nil, invalidAnswer("boolean answer has no probability")
	}

	return model.BooleanAnswer{
		Probability: *probability,
	}, nil
}

// ScoreProbabilities indexes a distribution keyed by level index. Every
// level must be below levelCount, the number of levels of the question, so
// that a malformed response cannot size the allocation. Levels left out are
// NaN, which callers reject as an incomplete distribution. It returns nil
// for a nil distribution.
func ScoreProbabilities(distribution map[string]float64, levelCount int) ([]float64, error) {
	if distribution == nil {
		return nil, nil
	}

	levels := make(map[int]float64, len(distribution))
	size := 0

	for key, probability := range distribution {
		level, err := strconv.Atoi(key)
		if err != nil || level < 0 || level >= levelCount {
			return nil, invalidAnswer(fmt.Sprintf("score probability level %q is not a level of the question", key))
		}

		levels[level] = probability
		size = max(size, level+1)
	}

	probabilities := make([]float64, size)
	for index := range probabilities {
		probabilities[index] = math.NaN()
	}

	for level, probability := range levels {
		probabilities[level] = probability
	}

	return probabilities, nil
}

// ScoreLevelCount returns the number of levels of a score question, and zero
// for other questions.
func ScoreLevelCount(question model.DecisionQuestion) int {
	score, ok := question.(model.ScoreQuestion)
	if !ok {
		return 0
	}

	return len(score.Levels)
}

// ScoreLevelCounts returns the number of levels of every score question,
// keyed by question ID.
func ScoreLevelCounts(questions []model.DecisionQuestion) map[string]int {
	counts := make(map[string]int, len(questions))

	for _, question := range questions {
		counts[question.QuestionID()] = ScoreLevelCount(question)
	}

	return counts
}

// ConvertAnswers converts answers keyed by question ID and checks that
// there is exactly one per question.
func ConvertAnswers(answers map[string]Answer, questions []model.DecisionQuestion, provider string) (map[string]model.DecisionAnswer, error) {
	if len(answers) != len(questions) {
		return nil, missingAnswers(provider)
	}

	converted := make(map[string]model.DecisionAnswer, len(answers))

	for _, question := range questions {
		answer, ok := answers[question.QuestionID()]
		if !ok {
			return nil, missingAnswers(provider)
		}

		value, err := answer.Convert(question)
		if err != nil {
			return nil, fmt.Errorf("answer to question %q: %w", question.QuestionID(), err)
		}

		converted[question.QuestionID()] = value
	}

	return converted, nil
}

// Confidence collects the confidence of the answers that report it, keyed
// by question ID.
func Confidence(answers map[string]Answer) map[string]float64 {
	confidence := make(map[string]float64)

	for questionID, answer := range answers {
		if answer.Confidence != nil {
			confidence[questionID] = *answer.Confidence
		}
	}

	return confidence
}

func missingAnswers(provider string) error {
	return fmt.Errorf("%w: %s must return exactly one answer for every question", model.ErrInvalidResponse, provider)
}

func invalidAnswer(message string) error {
	return fmt.Errorf("%w: %s", model.ErrInvalidResponse, message)
}
