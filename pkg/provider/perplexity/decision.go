package perplexity

import (
	"context"
	"encoding/json"
	"maps"
	"slices"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/decisionwire"
	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/model"
)

// decisionsPath is the Decisions API endpoint.
const decisionsPath = "/v1/decisions"

// decisionQuestionTypes name the question types on the wire. Boolean
// questions are Perplexity's "noul" questions.
var decisionQuestionTypes = decisionwire.QuestionTypeNames{
	model.DecisionQuestionChoice:  "choice",
	model.DecisionQuestionScore:   "score",
	model.DecisionQuestionBoolean: "noul",
}

var (
	_ model.DecisionModel    = (*DecisionModel)(nil)
	_ model.DecisionProvider = (*Provider)(nil)
)

// DecisionMetadata is the "perplexity" provider metadata of a decision.
type DecisionMetadata struct {
	// Confidence maps question IDs to the confidence of choice and score
	// answers. It is separate from the answers' probabilities.
	Confidence map[string]float64 `json:"confidence"`

	// Legend maps score question IDs to the legend Perplexity returned.
	Legend map[string]json.RawMessage `json:"legend"`
}

// Decision returns a Decisions API model, such as "pplx-decider-v1.2".
// Decision models are experimental.
func (p *Provider) Decision(modelID string) *DecisionModel {
	return &DecisionModel{
		modelID:  modelID,
		provider: p,
	}
}

// DecisionModel returns Decision(modelID).
func (p *Provider) DecisionModel(modelID string) (model.DecisionModel, error) {
	return p.Decision(modelID), nil
}

// DecisionModel is a Perplexity Decisions API model that answers choice,
// score and boolean questions. Perplexity reports full-precision
// probabilities. The decision contract is experimental and may change.
type DecisionModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "perplexity.decision" by
// default.
func (m *DecisionModel) Provider() string {
	return m.provider.name + ".decision"
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

// decisionRequest is the body of a Decisions API request.
type decisionRequest struct {
	Model     string                 `json:"model"`
	State     json.RawMessage        `json:"state"`
	Questions decisionwire.Questions `json:"questions"`
}

// decisionResponse is the body of a Decisions API response.
type decisionResponse struct {
	Model   *string                   `json:"model"`
	Answers map[string]decisionAnswer `json:"answers"`
	Usage   *decisionUsage            `json:"usage"`
}

// decisionAnswer is an answer with Perplexity's score legend.
type decisionAnswer struct {
	decisionwire.Answer

	Legend json.RawMessage `json:"legend"`
}

type decisionUsage struct {
	InputTokens  *int `json:"input_tokens"`
	OutputTokens *int `json:"output_tokens"`
}

// Decide answers every question with one Decisions API call.
func (m *DecisionModel) Decide(ctx context.Context, options model.DecisionOptions) (*model.DecisionResult, error) {
	questions, err := m.decisionQuestions(options.Questions)
	if err != nil {
		return nil, err
	}

	state, err := decisionState(options.State)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(decisionsPath, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, decisionRequest{
		Model:     m.modelID,
		State:     state,
		Questions: questions,
	})
	if err != nil {
		return nil, err
	}

	var decoded decisionResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	answers, err := convertDecisionAnswers(decoded.Answers, options.Questions)
	if err != nil {
		return nil, err
	}

	warnings, err := decisionWarnings(options)
	if err != nil {
		return nil, err
	}

	modelID := m.modelID
	if decoded.Model != nil {
		modelID = *decoded.Model
	}

	return &model.DecisionResult{
		Answers:  answers,
		Usage:    decoded.Usage.decisionUsage(),
		Warnings: warnings,
		ProviderMetadata: model.ProviderMetadata{
			Namespace: decisionMetadata(decoded.Answers),
		},
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				ID:        response.Headers.Get("X-Request-Id"),
				Timestamp: time.Now(),
				ModelID:   modelID,
			},
			Headers: response.Headers,
			Body:    response.Body,
		},
	}, nil
}

// decisionQuestions converts the questions, checking the limits of the
// Decisions API.
func (m *DecisionModel) decisionQuestions(questions []model.DecisionQuestion) (decisionwire.Questions, error) {
	if len(questions) == 0 || len(questions) > maxDecisionQuestions {
		return nil, invalidDecisionArgument("Perplexity Decisions supports 1 to 128 questions per request")
	}

	converted := make(decisionwire.Questions, 0, len(questions))

	for _, question := range questions {
		if err := validateDecisionQuestion(question); err != nil {
			return nil, err
		}

		encoded, supported, err := decisionwire.EncodeQuestion(question, decisionQuestionTypes)
		if err != nil {
			return nil, err
		}

		if !supported {
			return nil, &model.DecisionUnsupportedQuestionTypeError{
				QuestionID:   question.QuestionID(),
				QuestionType: question.QuestionType(),
				Provider:     m.Provider(),
				ModelID:      m.modelID,
			}
		}

		converted = append(converted, decisionwire.NamedQuestion{
			ID:       question.QuestionID(),
			Question: encoded,
		})
	}

	return converted, nil
}

// convertDecisionAnswers converts the answers and checks that there is
// exactly one per question.
func convertDecisionAnswers(answers map[string]decisionAnswer, questions []model.DecisionQuestion) (map[string]model.DecisionAnswer, error) {
	plain := make(map[string]decisionwire.Answer, len(answers))
	for questionID, answer := range answers {
		plain[questionID] = answer.Answer
	}

	return decisionwire.ConvertAnswers(plain, questions, "Perplexity Decisions")
}

// decisionMetadata collects the confidence of choice and score answers and
// the legends of score answers.
func decisionMetadata(answers map[string]decisionAnswer) DecisionMetadata {
	metadata := DecisionMetadata{
		Confidence: make(map[string]float64),
		Legend:     make(map[string]json.RawMessage),
	}

	for questionID, answer := range answers {
		if answer.Type != "noul" && answer.Confidence != nil {
			metadata.Confidence[questionID] = *answer.Confidence
		}

		if answer.Type == "score" && wirejson.HasValue(answer.Legend) {
			metadata.Legend[questionID] = answer.Legend
		}
	}

	return metadata
}

// decisionWarnings reports the settings the Decisions API does not accept.
func decisionWarnings(options model.DecisionOptions) ([]model.Warning, error) {
	var warnings []model.Warning

	if options.MaxOutputTokens != nil {
		warnings = append(warnings, model.UnsupportedWarning("maxOutputTokens", ""))
	}

	if options.Reasoning != "" {
		warnings = append(warnings, model.UnsupportedWarning("reasoning", ""))
	}

	perplexityOptions, _, err := model.ProviderValue[map[string]json.RawMessage](options.ProviderOptions, Namespace)
	if err != nil {
		return nil, err
	}

	for _, name := range slices.Sorted(maps.Keys(perplexityOptions)) {
		warnings = append(warnings, model.UnsupportedWarning("providerOptions.perplexity."+name, ""))
	}

	return warnings, nil
}

func (u *decisionUsage) decisionUsage() *model.DecisionUsage {
	if u == nil {
		return nil
	}

	return &model.DecisionUsage{
		InputTokens:  u.InputTokens,
		OutputTokens: u.OutputTokens,
	}
}
