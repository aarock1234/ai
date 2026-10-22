package typesafeai

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"net/http"
	"slices"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/decisionwire"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// decisionPath is the System One endpoint.
	decisionPath = "/systemone"

	// decisionRoundingDecimals is the precision of TypeSafe's probabilities
	// and scores.
	decisionRoundingDecimals = 2

	maxDecisionChoices = 255
	maxDecisionLevels  = 10
)

// decisionQuestionTypes name the question types on the wire. Boolean
// questions are TypeSafe's "noul" primitive.
var decisionQuestionTypes = decisionwire.QuestionTypeNames{
	model.DecisionQuestionChoice:  "choice",
	model.DecisionQuestionScore:   "score",
	model.DecisionQuestionBox:     "box",
	model.DecisionQuestionBoolean: "noul",
}

var _ model.DecisionModel = (*DecisionModel)(nil)

// DecisionOptions are the decision settings read from the "typesafe"
// provider options namespace.
type DecisionOptions struct {
	// LongContext enables TypeSafe's experimental long context mode, which
	// raises the request and question context limits to 262k tokens. It is
	// slower and less accurate than the default limits.
	LongContext *bool `json:"longContext,omitempty"`
}

// DecisionMetadata is the "typesafe" provider metadata of a decision.
type DecisionMetadata struct {
	// Confidence maps question IDs to TypeSafe's confidence, which is
	// separate from the answers' probabilities.
	Confidence map[string]float64 `json:"confidence"`
}

// DecisionModel is a TypeSafe System One model that answers choice, score,
// box and boolean questions in one request against the shared state.
// TypeSafe rounds probabilities and scores to two decimal places. The
// decision contract is experimental and may change.
type DecisionModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "typesafe.decision" by
// default.
func (m *DecisionModel) Provider() string {
	return m.provider.name + ".decision"
}

// ModelID returns the model identifier.
func (m *DecisionModel) ModelID() string {
	return m.modelID
}

// SupportedQuestionTypes returns choice, score, boolean and box.
func (m *DecisionModel) SupportedQuestionTypes() []model.DecisionQuestionType {
	return []model.DecisionQuestionType{
		model.DecisionQuestionChoice,
		model.DecisionQuestionScore,
		model.DecisionQuestionBoolean,
		model.DecisionQuestionBox,
	}
}

// decisionRequest is the body of a System One request.
type decisionRequest struct {
	Model     string                 `json:"model"`
	State     json.RawMessage        `json:"state"`
	Assets    map[string]imageAsset  `json:"assets,omitempty"`
	Questions decisionwire.Questions `json:"questions"`
}

// decisionResponse is the body of a System One response.
type decisionResponse struct {
	Model   *string                        `json:"model"`
	Answers map[string]decisionwire.Answer `json:"answers"`
	Usage   *decisionUsage                 `json:"usage"`
}

type decisionUsage struct {
	InputTokens  *int `json:"input_tokens"`
	OutputTokens *int `json:"output_tokens"`
}

// Decide answers every question with one System One call. Requests with
// images are sent as multipart forms.
func (m *DecisionModel) Decide(ctx context.Context, options model.DecisionOptions) (*model.DecisionResult, error) {
	questions, err := m.decisionQuestions(options.Questions)
	if err != nil {
		return nil, err
	}

	typesafeOptions, warnings, err := readDecisionOptions(options)
	if err != nil {
		return nil, err
	}

	state, err := prepareDecisionState(options.State, m.provider.generateID)
	if err != nil {
		return nil, err
	}

	body := decisionRequest{
		Model:     m.modelID,
		State:     state.state,
		Questions: questions,
	}

	if len(state.assets) > 0 {
		body.Assets = state.assets
	}

	request, err := m.provider.apiRequest(decisionPath, featureHeaders(state, options.Questions, typesafeOptions), options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := sendDecision(ctx, request, body, state.images)
	if err != nil {
		return nil, err
	}

	var decoded decisionResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	answers, err := decisionwire.ConvertAnswers(decoded.Answers, options.Questions, "TypeSafe")
	if err != nil {
		return nil, err
	}

	modelID := m.modelID
	if decoded.Model != nil {
		modelID = *decoded.Model
	}

	return &model.DecisionResult{
		Answers: answers,
		Rounding: &model.DecisionRounding{
			ProbabilityDecimals: new(decisionRoundingDecimals),
			ScoreDecimals:       new(decisionRoundingDecimals),
		},
		Usage:    decoded.Usage.decisionUsage(),
		Warnings: warnings,
		ProviderMetadata: model.ProviderMetadata{
			Namespace: DecisionMetadata{
				Confidence: decisionwire.Confidence(decoded.Answers),
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

// decisionQuestions converts the questions, checking TypeSafe's limits.
func (m *DecisionModel) decisionQuestions(questions []model.DecisionQuestion) (decisionwire.Questions, error) {
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

func validateDecisionQuestion(question model.DecisionQuestion) error {
	switch typed := question.(type) {
	case model.ChoiceQuestion:
		if len(typed.Choices) > maxDecisionChoices {
			return fmt.Errorf("%w: questions.%s.criteria: TypeSafe Choice questions support at most 255 options", model.ErrInvalidArgument, typed.ID)
		}

	case model.ScoreQuestion:
		if len(typed.Levels) > maxDecisionLevels {
			return fmt.Errorf("%w: questions.%s.criteria: TypeSafe Score questions support at most 10 levels", model.ErrInvalidArgument, typed.ID)
		}
	}

	return nil
}

// readDecisionOptions reads the options and warns about the settings
// TypeSafe does not accept.
func readDecisionOptions(options model.DecisionOptions) (DecisionOptions, []model.Warning, error) {
	typesafeOptions, _, err := model.ProviderValue[DecisionOptions](options.ProviderOptions, Namespace)
	if err != nil {
		return DecisionOptions{}, nil, err
	}

	names, _, err := model.ProviderValue[map[string]json.RawMessage](options.ProviderOptions, Namespace)
	if err != nil {
		return DecisionOptions{}, nil, err
	}

	var warnings []model.Warning

	for _, name := range slices.Sorted(maps.Keys(names)) {
		if name != "longContext" {
			warnings = append(warnings, model.UnsupportedWarning("providerOptions.typesafe."+name, ""))
		}
	}

	if options.MaxOutputTokens != nil {
		warnings = append(warnings, model.UnsupportedWarning("maxOutputTokens", ""))
	}

	if options.Reasoning != "" {
		warnings = append(warnings, model.UnsupportedWarning("reasoning", ""))
	}

	return typesafeOptions, warnings, nil
}

// featureHeaders opt in to the image and long context features the request
// uses.
func featureHeaders(state *preparedState, questions []model.DecisionQuestion, options DecisionOptions) http.Header {
	headers := make(http.Header)

	hasBoxQuestion := slices.ContainsFunc(questions, func(question model.DecisionQuestion) bool {
		return question.QuestionType() == model.DecisionQuestionBox
	})

	if len(state.images) > 0 || hasBoxQuestion {
		headers.Set("TypeSafe-Danger-Zone-Images", "true")
	}

	if options.LongContext != nil && *options.LongContext {
		headers.Set("TypeSafe-Danger-Zone-Long-Context", "true")
	}

	return headers
}

// sendDecision posts the request as JSON, or as a multipart form with the
// request JSON and the images when there are images.
func sendDecision(ctx context.Context, request apicall.Request, body decisionRequest, images []imagePart) (*apicall.Response, error) {
	if len(images) == 0 {
		return apicall.PostJSON(ctx, request, body)
	}

	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("encode request body: %w", err)
	}

	fields := []apicall.FormField{
		{
			Name:      "request",
			Filename:  "request.json",
			MediaType: "application/json",
			Data:      encoded,
		},
	}

	for _, image := range images {
		fields = append(fields, apicall.FormField{
			Name:      image.name,
			Filename:  image.filename,
			MediaType: image.mediaType,
			Data:      image.data,
		})
	}

	return apicall.PostForm(ctx, request, fields)
}

func (u *decisionUsage) decisionUsage() *model.DecisionUsage {
	if u == nil {
		return &model.DecisionUsage{}
	}

	return &model.DecisionUsage{
		InputTokens:  u.InputTokens,
		OutputTokens: u.OutputTokens,
	}
}
