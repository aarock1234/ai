package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/internal/decisionwire"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/pkg/model"
)

// decisionMaxSafetyIdentifierLength bounds the safety identifier.
const decisionMaxSafetyIdentifierLength = 128

// decisionRoundingDecimals is the precision of the Decisions API's
// probabilities and scores.
const decisionRoundingDecimals = 2

// decisionImageMediaTypes are the image types the Decisions API accepts.
var decisionImageMediaTypes = []string{
	"image/png",
	"image/jpeg",
	"image/webp",
	"image/gif",
}

// DecisionOptions are the decision settings read from the "openai"
// provider options namespace.
type DecisionOptions struct {
	// SafetyIdentifier is an opaque end-user identifier for safety
	// monitoring, up to 128 characters.
	SafetyIdentifier string `json:"safetyIdentifier,omitempty"`
}

// DecisionMetadata is the "openai" provider metadata of a decision.
type DecisionMetadata struct {
	// Usage is the API's usage object, if reported.
	Usage json.RawMessage `json:"usage,omitempty"`

	// Confidence maps question IDs to the confidence the API reported for
	// choice and score answers.
	Confidence map[string]float64 `json:"confidence"`
}

// DecisionModel is an OpenAI Decisions API model, such as gpt-6-luna,
// that answers choice, score and boolean questions. The decision contract
// is experimental and may change.
type DecisionModel struct {
	modelID string
	config  config
}

func newDecisionModel(modelID string, c config) *DecisionModel {
	return &DecisionModel{
		modelID: modelID,
		config:  c,
	}
}

// Provider returns the provider identifier, such as "openai.decision".
func (m *DecisionModel) Provider() string {
	return m.config.modelProvider("decision")
}

// ModelID returns the model identifier.
func (m *DecisionModel) ModelID() string {
	return m.modelID
}

// SupportedQuestionTypes returns choice, score and boolean.
func (m *DecisionModel) SupportedQuestionTypes() []model.DecisionQuestionType {
	return []model.DecisionQuestionType{
		model.DecisionQuestionChoice,
		model.DecisionQuestionScore,
		model.DecisionQuestionBoolean,
	}
}

// Decide answers the questions with one Decisions API call.
func (m *DecisionModel) Decide(ctx context.Context, options model.DecisionOptions) (*model.DecisionResult, error) {
	openaiOptions, err := readDecisionOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	warnings, err := decisionWarnings(options)
	if err != nil {
		return nil, err
	}

	body, err := m.decisionRequest(options, openaiOptions)
	if err != nil {
		return nil, err
	}

	request, err := m.config.request(ctx, "/decisions", options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, err
	}

	var decoded decisionResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	answers, err := decisionAnswers(decoded.Answers, options.Questions)
	if err != nil {
		return nil, err
	}

	modelID := decoded.Model
	if modelID == "" {
		modelID = m.modelID
	}

	return &model.DecisionResult{
		Answers: answers,
		Usage:   decoded.Usage.decisionUsage(),
		Rounding: &model.DecisionRounding{
			ProbabilityDecimals: new(decisionRoundingDecimals),
			ScoreDecimals:       new(decisionRoundingDecimals),
		},
		Warnings: warnings,
		ProviderMetadata: model.ProviderMetadata{
			Namespace: DecisionMetadata{
				Usage:      decoded.Usage.rawUsage(),
				Confidence: decisionConfidence(decoded.Answers),
			},
		},
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: time.Now(),
				ModelID:   modelID,
			},
			Headers: response.Headers,
			Body:    response.Body,
		},
	}, nil
}

func readDecisionOptions(providerOptions model.ProviderOptions) (DecisionOptions, error) {
	options, _, err := model.ProviderValue[DecisionOptions](providerOptions, Namespace)
	if err != nil {
		return DecisionOptions{}, err
	}

	if len(options.SafetyIdentifier) > decisionMaxSafetyIdentifierLength {
		return DecisionOptions{}, fmt.Errorf(
			"%w: providerOptions.openai.safetyIdentifier must be at most %d characters",
			model.ErrInvalidArgument,
			decisionMaxSafetyIdentifierLength,
		)
	}

	return options, nil
}

// decisionWarnings reports settings the Decisions API does not accept.
func decisionWarnings(options model.DecisionOptions) ([]model.Warning, error) {
	var warnings []model.Warning

	if options.MaxOutputTokens != nil {
		warnings = append(warnings, model.UnsupportedWarning("maxOutputTokens", ""))
	}

	if options.Reasoning != "" {
		warnings = append(warnings, model.UnsupportedWarning("reasoning", ""))
	}

	openaiOptions, _, err := model.ProviderValue[map[string]json.RawMessage](options.ProviderOptions, Namespace)
	if err != nil {
		return nil, err
	}

	for _, name := range slices.Sorted(maps.Keys(openaiOptions)) {
		if name == "safetyIdentifier" {
			continue
		}

		warnings = append(warnings, model.UnsupportedWarning("providerOptions.openai."+name, ""))
	}

	return warnings, nil
}

// decisionRequest converts the state and the questions.
func (m *DecisionModel) decisionRequest(options model.DecisionOptions, openaiOptions DecisionOptions) (decisionRequestBody, error) {
	content := make([]decisionContent, 0, len(options.State))

	for _, part := range options.State {
		converted, err := decisionStateContent(part)
		if err != nil {
			return decisionRequestBody{}, err
		}

		content = append(content, converted)
	}

	questions := make([]decisionQuestion, 0, len(options.Questions))

	for _, question := range options.Questions {
		converted, err := m.convertQuestion(question)
		if err != nil {
			return decisionRequestBody{}, err
		}

		questions = append(questions, converted)
	}

	return decisionRequestBody{
		Model:            m.modelID,
		SafetyIdentifier: openaiOptions.SafetyIdentifier,
		Input: []decisionInputMessage{
			{
				Role:    "user",
				Content: content,
			},
		},
		Questions: questions,
	}, nil
}

// decisionStateContent converts one state part. JSON is sent as text, and
// files must be inline images.
func decisionStateContent(part model.DecisionStatePart) (decisionContent, error) {
	switch typed := part.(type) {
	case model.TextPart:
		return decisionContent{
			Type: "input_text",
			Text: typed.Text,
		}, nil

	case model.DecisionJSONPart:
		return decisionContent{
			Type: "input_text",
			Text: model.DecisionInputText(model.DecisionJSON(typed.Value)),
		}, nil

	case model.FilePart:
		return decisionImageContent(typed)

	default:
		return decisionContent{}, fmt.Errorf("%w: decision state part %T", model.ErrUnsupported, part)
	}
}

func decisionImageContent(part model.FilePart) (decisionContent, error) {
	data, ok := part.Data.(model.FileBytes)
	if !ok {
		return decisionContent{}, fmt.Errorf(
			"%w: OpenAI decision file input: %s (%T)",
			model.ErrUnsupported,
			part.MediaType,
			part.Data,
		)
	}

	mediaType := part.MediaType
	if !mediatype.IsFull(mediaType) {
		mediaType, _ = mediatype.Detect(data.Data, "image")
	}

	if !slices.Contains(decisionImageMediaTypes, mediaType) {
		return decisionContent{}, fmt.Errorf(
			"%w: OpenAI decision image media type: %s",
			model.ErrUnsupported,
			part.MediaType,
		)
	}

	partOptions, _, err := model.ProviderValue[ChatPartOptions](part.ProviderOptions, Namespace)
	if err != nil {
		return decisionContent{}, err
	}

	return decisionContent{
		Type:     "input_image",
		ImageURL: dataurl.Encode(mediaType, data.Data),
		Detail:   partOptions.ImageDetail,
	}, nil
}

// convertQuestion converts a question. Boolean questions are
// predicates whose instructions include the criteria.
func (m *DecisionModel) convertQuestion(question model.DecisionQuestion) (decisionQuestion, error) {
	switch typed := question.(type) {
	case model.BooleanQuestion:
		return decisionQuestion{
			Type:         "predicate",
			Name:         typed.ID,
			Instructions: new(predicateInstructions(typed)),
		}, nil

	case model.ChoiceQuestion:
		return decisionQuestion{
			Type:         "choice",
			Name:         typed.ID,
			Instructions: decisionInstructions(typed.Instructions),
			Choices:      decisionChoices(typed.Choices),
		}, nil

	case model.ScoreQuestion:
		return decisionQuestion{
			Type:         "score",
			Name:         typed.ID,
			Instructions: decisionInstructions(typed.Instructions),
			Levels:       decisionLevels(typed.Levels),
		}, nil

	default:
		return decisionQuestion{}, &model.DecisionUnsupportedQuestionTypeError{
			QuestionID:   question.QuestionID(),
			QuestionType: question.QuestionType(),
			Provider:     m.Provider(),
			ModelID:      m.modelID,
		}
	}
}

// predicateInstructions joins the instructions and the criteria of a
// boolean question.
func predicateInstructions(question model.BooleanQuestion) string {
	var sections []string

	if question.Instructions != nil {
		sections = append(sections, model.DecisionInputText(question.Instructions))
	}

	if question.True != nil {
		sections = append(sections, "Criteria for true:\n"+model.DecisionInputText(question.True))
	}

	if question.False != nil {
		sections = append(sections, "Criteria for false:\n"+model.DecisionInputText(question.False))
	}

	return strings.Join(sections, "\n\n")
}

func decisionInstructions(instructions model.DecisionInput) *string {
	if instructions == nil {
		return nil
	}

	return new(model.DecisionInputText(instructions))
}

func decisionChoices(choices []model.DecisionChoice) []decisionChoice {
	converted := make([]decisionChoice, len(choices))

	for index, choice := range choices {
		converted[index] = decisionChoice{
			Value:       choice.Name,
			Description: model.DecisionInputText(choice.Description),
		}
	}

	return converted
}

// decisionLevels labels score levels with their indices, because score
// criteria have no separate labels.
func decisionLevels(levels []model.DecisionInput) []decisionLevel {
	converted := make([]decisionLevel, len(levels))

	for index, level := range levels {
		converted[index] = decisionLevel{
			Label:       strconv.Itoa(index),
			Description: model.DecisionInputText(level),
		}
	}

	return converted
}

// decisionAnswers converts the answers and checks that there is exactly
// one per question.
func decisionAnswers(answers []decisionAnswer, questions []model.DecisionQuestion) (map[string]model.DecisionAnswer, error) {
	converted := make(map[string]model.DecisionAnswer, len(answers))
	levelCounts := decisionwire.ScoreLevelCounts(questions)

	for _, answer := range answers {
		name, value, err := answer.convert(levelCounts)
		if err != nil {
			return nil, err
		}

		if _, duplicate := converted[name]; duplicate || !hasDecisionQuestion(questions, name) {
			return nil, invalidDecisionResponse("decisions must return exactly one answer for every question")
		}

		converted[name] = value
	}

	if len(converted) != len(questions) {
		return nil, invalidDecisionResponse("decisions must return exactly one answer for every question")
	}

	return converted, nil
}

func hasDecisionQuestion(questions []model.DecisionQuestion, name string) bool {
	return slices.ContainsFunc(questions, func(question model.DecisionQuestion) bool {
		return question.QuestionID() == name
	})
}

// decisionConfidence collects the confidence of choice and score answers.
func decisionConfidence(answers []decisionAnswer) map[string]float64 {
	confidence := make(map[string]float64)

	for _, answer := range answers {
		if answer.Confidence != nil && (answer.Type == "choice" || answer.Type == "score") {
			confidence[answer.Name] = *answer.Confidence
		}
	}

	return confidence
}

// convert returns the question name and the normalized answer. levelCounts
// bounds the levels of score probabilities by question name.
func (a decisionAnswer) convert(levelCounts map[string]int) (string, model.DecisionAnswer, error) {
	switch a.Type {
	case "refusal":
		if a.RefusalName == nil {
			return "", nil, invalidDecisionResponse("OpenAI Decisions refused an unnamed question")
		}

		return *a.RefusalName, model.RefusalAnswer{}, nil

	case "predicate":
		if a.Probability == nil || !isUnitInterval(*a.Probability) {
			return "", nil, invalidDecisionResponse("predicate answer has no valid probability")
		}

		return a.Name, model.BooleanAnswer{
			Probability: *a.Probability,
		}, nil

	case "choice":
		probabilities, err := a.choiceProbabilities()
		if err != nil {
			return "", nil, err
		}

		return a.Name, model.ChoiceAnswer{
			Choice:        a.Choice,
			Probabilities: probabilities,
		}, nil

	case "score":
		probabilities, err := a.scoreProbabilities(levelCounts[a.Name])
		if err != nil {
			return "", nil, err
		}

		return a.Name, model.ScoreAnswer{
			Score:         a.Score,
			Probabilities: probabilities,
		}, nil

	default:
		return "", nil, invalidDecisionResponse(fmt.Sprintf("unknown answer type %q", a.Type))
	}
}

func (a decisionAnswer) choiceProbabilities() (map[string]float64, error) {
	probabilities := make(map[string]float64, len(a.ChoiceProbabilities))

	for _, entry := range a.ChoiceProbabilities {
		if !isUnitInterval(entry.Probability) {
			return nil, invalidDecisionResponse("choice probabilities must be in [0, 1]")
		}

		if _, duplicate := probabilities[entry.Value]; duplicate {
			return nil, invalidDecisionResponse("decisions returned duplicate probability values")
		}

		probabilities[entry.Value] = entry.Probability
	}

	return probabilities, nil
}

// scoreProbabilities indexes the distribution by level. Levels must be below
// levelCount, the number of levels of the question, so that a malformed
// response cannot size the allocation. Levels the API left out are NaN,
// which the caller rejects as an incomplete distribution.
func (a decisionAnswer) scoreProbabilities(levelCount int) ([]float64, error) {
	size := 0
	for _, entry := range a.ScoreProbabilities {
		if entry.Value < 0 || entry.Value >= levelCount {
			return nil, invalidDecisionResponse(fmt.Sprintf("score probability level %d is not a level of the question", entry.Value))
		}

		if !isUnitInterval(entry.Probability) {
			return nil, invalidDecisionResponse("score probabilities must be in [0, 1]")
		}

		size = max(size, entry.Value+1)
	}

	probabilities := make([]float64, size)
	for index := range probabilities {
		probabilities[index] = math.NaN()
	}

	for _, entry := range a.ScoreProbabilities {
		if !math.IsNaN(probabilities[entry.Value]) {
			return nil, invalidDecisionResponse("decisions returned duplicate probability values")
		}

		probabilities[entry.Value] = entry.Probability
	}

	return probabilities, nil
}

func isUnitInterval(value float64) bool {
	return value >= 0 && value <= 1
}

func invalidDecisionResponse(message string) error {
	return fmt.Errorf("%w: %s", model.ErrInvalidResponse, message)
}
