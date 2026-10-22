// Package decisionlm adapts a language model with structured output into a
// decision model that answers choice, score and boolean questions. It is a
// port of the AI SDK's DecisionLanguageModel.
package decisionlm

import (
	_ "embed"

	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/internal/decisionwire"
	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/pkg/model"
)

// systemPrompt instructs the language model how to answer.
//
//go:embed system_prompt.md
var systemPrompt string

var _ model.DecisionModel = (*Model)(nil)

// Model is a decision model backed by a language model. Each decision is
// one structured-output call.
type Model struct {
	languageModel model.LanguageModel
	provider      string
}

// New returns a decision model backed by languageModel. An empty provider
// reports the language model's provider with a ".decision" suffix.
func New(languageModel model.LanguageModel, provider string) *Model {
	if provider == "" {
		provider = languageModel.Provider() + ".decision"
	}

	return &Model{
		languageModel: languageModel,
		provider:      provider,
	}
}

// Provider returns the provider identifier.
func (m *Model) Provider() string {
	return m.provider
}

// ModelID returns the language model's ID.
func (m *Model) ModelID() string {
	return m.languageModel.ModelID()
}

// SupportedQuestionTypes returns choice, score and boolean.
func (m *Model) SupportedQuestionTypes() []model.DecisionQuestionType {
	return []model.DecisionQuestionType{
		model.DecisionQuestionChoice,
		model.DecisionQuestionScore,
		model.DecisionQuestionBoolean,
	}
}

// Decide answers every question with one language model call whose JSON
// schema has one property per question. Questions are keyed internally as
// q0, q1 and so on, and choices as c0, c1 and so on, which avoids schema
// restrictions on caller IDs and case-sensitive labels.
func (m *Model) Decide(ctx context.Context, options model.DecisionOptions) (*model.DecisionResult, error) {
	if ctx.Err() != nil {
		return nil, context.Cause(ctx)
	}

	if err := m.validate(options); err != nil {
		return nil, err
	}

	callOptions, err := m.callOptions(options)
	if err != nil {
		return nil, err
	}

	result, err := m.languageModel.Generate(ctx, callOptions)
	if err != nil {
		return nil, err
	}

	if ctx.Err() != nil {
		return nil, context.Cause(ctx)
	}

	if result.FinishReason.Unified != model.FinishStop {
		return nil, fmt.Errorf("%w: decision did not complete: %s", model.ErrInvalidResponse, result.FinishReason.Unified)
	}

	answers, err := decodeAnswers(result.Content, options.Questions)
	if err != nil {
		return nil, err
	}

	return &model.DecisionResult{
		Answers: answers,
		Usage: &model.DecisionUsage{
			InputTokens:  result.Usage.InputTokens.Total,
			OutputTokens: result.Usage.OutputTokens.Total,
		},
		Warnings:         result.Warnings,
		ProviderMetadata: result.ProviderMetadata,
		Response:         result.Response,
	}, nil
}

// validate checks the whole call before any request is built: image-only
// files, supported question types, and choices and levels to choose from.
func (m *Model) validate(options model.DecisionOptions) error {
	for _, part := range options.State {
		file, ok := part.(model.FilePart)
		if ok && mediatype.TopLevel(file.MediaType) != "image" {
			return fmt.Errorf("%w: decision file media type: %s", model.ErrUnsupported, file.MediaType)
		}
	}

	for _, question := range options.Questions {
		if !isSupportedQuestion(question) {
			return &model.DecisionUnsupportedQuestionTypeError{
				QuestionID:   question.QuestionID(),
				QuestionType: question.QuestionType(),
				Provider:     m.provider,
				ModelID:      m.ModelID(),
			}
		}
	}

	if len(options.Questions) == 0 {
		return fmt.Errorf("%w: decision requires at least one question", model.ErrInvalidArgument)
	}

	for _, question := range options.Questions {
		if !hasEnoughCriteria(question) {
			return fmt.Errorf(
				"%w: questions.%s.criteria: Choice requires at least one option; Score requires at least two levels",
				model.ErrInvalidArgument,
				question.QuestionID(),
			)
		}
	}

	return nil
}

// isSupportedQuestion reports whether the answer schema can express the
// question: choice, score and boolean questions are supported.
func isSupportedQuestion(question model.DecisionQuestion) bool {
	switch question.(type) {
	case model.ChoiceQuestion, model.ScoreQuestion, model.BooleanQuestion:
		return true
	default:
		return false
	}
}

func hasEnoughCriteria(question model.DecisionQuestion) bool {
	switch typed := question.(type) {
	case model.ChoiceQuestion:
		return len(typed.Choices) > 0

	case model.ScoreQuestion:
		return len(typed.Levels) >= 2

	default:
		return true
	}
}

// callOptions builds the structured-output call.
func (m *Model) callOptions(options model.DecisionOptions) (model.CallOptions, error) {
	rubrics, err := encodeRubrics(options.Questions)
	if err != nil {
		return model.CallOptions{}, err
	}

	schema, err := answerSchema(options.Questions)
	if err != nil {
		return model.CallOptions{}, err
	}

	content := []model.UserPart{
		model.TextPart{
			Text: string(rubrics),
		},
		model.TextPart{
			Text: "Shared state:",
		},
	}

	for _, part := range options.State {
		content = append(content, statePart(part))
	}

	reasoningEffort := options.Reasoning
	if reasoningEffort == "" {
		reasoningEffort = model.ReasoningNone
	}

	return model.CallOptions{
		Prompt: []model.Message{
			model.SystemMessage{
				Content: strings.TrimSpace(systemPrompt),
			},
			model.UserMessage{
				Content: content,
			},
		},
		MaxOutputTokens: options.MaxOutputTokens,
		Reasoning:       reasoningEffort,
		ResponseFormat: &model.ResponseFormat{
			Type:   model.ResponseFormatJSON,
			Name:   "decision",
			Schema: schema,
		},
		Headers:         options.Headers,
		ProviderOptions: options.ProviderOptions,
	}, nil
}

// statePart converts a state part into prompt content. JSON parts become
// compact JSON text.
func statePart(part model.DecisionStatePart) model.UserPart {
	switch typed := part.(type) {
	case model.TextPart:
		return typed

	case model.FilePart:
		return typed

	case model.DecisionJSONPart:
		return model.TextPart{
			Text: model.DecisionInputText(model.DecisionJSON(typed.Value)),
		}

	default:
		return model.TextPart{}
	}
}

// questionKey is the internal key of the question at index.
func questionKey(index int) string {
	return "q" + strconv.Itoa(index)
}

// choiceKey is the internal code of the choice at index.
func choiceKey(index int) string {
	return "c" + strconv.Itoa(index)
}

// rubricQuestions holds the rubrics keyed by internal question key.
type rubricQuestions struct {
	Questions jsonobject.Object `json:"questions"`
}

// rubric is one question as the model reads it.
type rubric struct {
	ID           string          `json:"id"`
	Type         string          `json:"type"`
	Instructions json.RawMessage `json:"instructions,omitempty"`
	Criteria     json.RawMessage `json:"criteria,omitempty"`
}

// choiceRubric is one labeled option of a choice rubric.
type choiceRubric struct {
	Label       string          `json:"label"`
	Description json.RawMessage `json:"description"`
}

// encodeRubrics encodes the questions with their instructions and
// criteria. Choice options are keyed by internal code.
func encodeRubrics(questions []model.DecisionQuestion) (json.RawMessage, error) {
	var encoded jsonobject.Object

	for index, question := range questions {
		value, err := encodeRubric(question)
		if err != nil {
			return nil, err
		}

		encoded = encoded.With(questionKey(index), value)
	}

	rubrics, err := json.Marshal(rubricQuestions{
		Questions: encoded,
	})
	if err != nil {
		return nil, fmt.Errorf("encode decision rubrics: %w", err)
	}

	return rubrics, nil
}

func encodeRubric(question model.DecisionQuestion) (json.RawMessage, error) {
	wire, _, err := decisionwire.EncodeQuestion(question, decisionwire.NormalizedQuestionTypes)
	if err != nil {
		return nil, err
	}

	encoded := rubric{
		ID:           question.QuestionID(),
		Type:         wire.Type,
		Instructions: wire.Instructions,
		Criteria:     wire.Criteria,
	}

	if choice, ok := question.(model.ChoiceQuestion); ok {
		criteria, err := choiceCriteria(choice.Choices)
		if err != nil {
			return nil, err
		}

		encoded.Criteria = criteria
	}

	return json.Marshal(encoded)
}

func choiceCriteria(choices []model.DecisionChoice) (json.RawMessage, error) {
	var criteria jsonobject.Object

	for index, choice := range choices {
		value, err := json.Marshal(choiceRubric{
			Label:       choice.Name,
			Description: decisionwire.Input(choice.Description),
		})
		if err != nil {
			return nil, err
		}

		criteria = criteria.With(choiceKey(index), value)
	}

	return criteria.Encode()
}

// answerSchema returns the JSON schema with one required property per
// question.
func answerSchema(questions []model.DecisionQuestion) (json.RawMessage, error) {
	var properties jsonobject.Object

	required := make([]string, 0, len(questions))

	for index, question := range questions {
		property, err := answerProperty(question)
		if err != nil {
			return nil, err
		}

		properties = properties.With(questionKey(index), property)
		required = append(required, questionKey(index))
	}

	return json.Marshal(answerObjectSchema{
		Type:                 "object",
		Properties:           properties,
		Required:             required,
		AdditionalProperties: false,
	})
}

// answerObjectSchema is the schema of the answers object.
type answerObjectSchema struct {
	Type                 string            `json:"type"`
	Properties           jsonobject.Object `json:"properties"`
	Required             []string          `json:"required"`
	AdditionalProperties bool              `json:"additionalProperties"`
}

// answerPropertySchema is the schema of one answer.
type answerPropertySchema struct {
	Type        string   `json:"type"`
	Enum        []string `json:"enum,omitempty"`
	Description string   `json:"description,omitempty"`
}

func answerProperty(question model.DecisionQuestion) (json.RawMessage, error) {
	switch typed := question.(type) {
	case model.ChoiceQuestion:
		codes := make([]string, len(typed.Choices))
		for index := range typed.Choices {
			codes[index] = choiceKey(index)
		}

		return json.Marshal(answerPropertySchema{
			Type: "string",
			Enum: codes,
		})

	case model.ScoreQuestion:
		return json.Marshal(answerPropertySchema{
			Type: "number",
			Description: fmt.Sprintf(
				"A finite fractional score from 0 to %d, inclusive. Ordered rubric levels are indexed from zero.",
				len(typed.Levels)-1,
			),
		})

	default:
		return json.Marshal(answerPropertySchema{
			Type:        "number",
			Description: "Estimated probability that the answer is true, from 0 to 1 inclusive. 0 means certainly false and 1 means certainly true.",
		})
	}
}
