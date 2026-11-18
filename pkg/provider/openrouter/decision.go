package openrouter

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/decisionwire"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openaicompatible"
)

const (
	// decisionPath is the Decisions endpoint, relative to the API root
	// that holds the "v1" base URL.
	decisionPath = "/alpha/decisions"

	// decisionRoundingDecimals is the precision of OpenRouter's
	// probabilities and scores.
	decisionRoundingDecimals = 4
)

// decisionQuestionTypes name the question types on the wire. Boolean
// questions are the Decisions API's "noul".
var decisionQuestionTypes = decisionwire.QuestionTypeNames{
	model.DecisionQuestionChoice:  "choice",
	model.DecisionQuestionScore:   "score",
	model.DecisionQuestionBox:     "box",
	model.DecisionQuestionBoolean: "noul",
}

var (
	_ model.DecisionProvider = (*Provider)(nil)
	_ model.DecisionModel    = (*DecisionModel)(nil)
)

// DecisionOptions are the decision settings read from the "openrouter"
// provider options namespace.
type DecisionOptions struct {
	// Provider configures how OpenRouter routes the request to upstream
	// providers.
	Provider *ProviderRouting `json:"provider,omitempty"`

	// SessionID groups related requests in OpenRouter's logs.
	SessionID string `json:"session_id,omitempty"`

	// User identifies the end user, to help OpenRouter detect abuse.
	User string `json:"user,omitempty"`
}

// DecisionModel is a model of OpenRouter's Decisions API, such as
// "scaledown/classify-1" or "typesafe/jev-1.13". It answers choice, score,
// boolean and box questions about a text or JSON state in one request,
// with a probability for every outcome. Which question types a model
// serves depends on its provider. OpenRouter rounds probabilities and
// scores to four decimal places. The Decisions API is in alpha, and the
// decision contract is experimental.
//
// To decide with a chat model instead, adapt one with
// ai.NewDecisionLanguageModel(provider.Chat(modelID), "").
type DecisionModel struct {
	modelID  string
	provider *Provider
}

// Decision returns a Decisions API model.
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

// Provider returns the provider identifier, such as "openrouter.decision".
func (m *DecisionModel) Provider() string {
	return m.provider.name + ".decision"
}

// ModelID returns the model identifier.
func (m *DecisionModel) ModelID() string {
	return m.modelID
}

// SupportedQuestionTypes returns choice, score, boolean and box. A model
// whose provider serves fewer refuses the others with status 400.
func (m *DecisionModel) SupportedQuestionTypes() []model.DecisionQuestionType {
	return []model.DecisionQuestionType{
		model.DecisionQuestionChoice,
		model.DecisionQuestionScore,
		model.DecisionQuestionBoolean,
		model.DecisionQuestionBox,
	}
}

// decisionRequest is the body of a Decisions request.
type decisionRequest struct {
	Model     string                 `json:"model"`
	State     json.RawMessage        `json:"state"`
	Questions decisionwire.Questions `json:"questions"`
	Provider  *ProviderRouting       `json:"provider,omitempty"`
	SessionID string                 `json:"session_id,omitempty"`
	User      string                 `json:"user,omitempty"`
}

// decisionResponse is the body of a Decisions response.
type decisionResponse struct {
	Model    *string                        `json:"model"`
	Provider string                         `json:"provider"`
	Answers  map[string]decisionwire.Answer `json:"answers"`
	Usage    *decisionUsageWire             `json:"usage"`
}

// decisionUsageWire is a Decisions response's usage object.
type decisionUsageWire struct {
	InputTokens  *int         `json:"input_tokens"`
	OutputTokens *int         `json:"output_tokens"`
	Cost         *float64     `json:"cost"`
	IsBYOK       *bool        `json:"is_byok"`
	CostDetails  *costDetails `json:"cost_details"`
}

// Decide answers every question with one Decisions API request.
func (m *DecisionModel) Decide(ctx context.Context, options model.DecisionOptions) (*model.DecisionResult, error) {
	questions, err := m.decisionQuestions(options.Questions)
	if err != nil {
		return nil, err
	}

	decisionOptions, warnings, err := readDecisionOptions(options)
	if err != nil {
		return nil, err
	}

	state, err := decisionState(options.State)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.decisionRequest(options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, decisionRequest{
		Model:     m.modelID,
		State:     state,
		Questions: questions,
		Provider:  decisionOptions.Provider,
		SessionID: decisionOptions.SessionID,
		User:      decisionOptions.User,
	})
	if err != nil {
		return nil, err
	}

	var decoded decisionResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	answers, err := decisionwire.ConvertAnswers(decoded.Answers, options.Questions, "OpenRouter decisions")
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
		Usage:            decoded.Usage.decisionUsage(),
		Warnings:         warnings,
		ProviderMetadata: resultMetadata(decoded.Provider, decoded.Usage.accounting()),
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

// decisionQuestions converts the questions to the Decisions API's form.
func (m *DecisionModel) decisionQuestions(questions []model.DecisionQuestion) (decisionwire.Questions, error) {
	converted := make(decisionwire.Questions, 0, len(questions))

	for _, question := range questions {
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

// readDecisionOptions reads the options and warns about the settings the
// Decisions API does not accept.
func readDecisionOptions(options model.DecisionOptions) (DecisionOptions, []model.Warning, error) {
	decisionOptions, _, err := model.ProviderValue[DecisionOptions](options.ProviderOptions, Namespace)
	if err != nil {
		return DecisionOptions{}, nil, err
	}

	if decisionOptions.Provider != nil {
		if err := decisionOptions.Provider.validate(); err != nil {
			return DecisionOptions{}, nil, err
		}
	}

	var warnings []model.Warning

	if options.MaxOutputTokens != nil {
		warnings = append(warnings, model.UnsupportedWarning("maxOutputTokens", ""))
	}

	if options.Reasoning != "" {
		warnings = append(warnings, model.UnsupportedWarning("reasoning", ""))
	}

	return decisionOptions, warnings, nil
}

// decisionState encodes the state: one text part as a string, and several
// parts, or JSON, as an array of strings and JSON values. Files are not
// supported.
func decisionState(parts []model.DecisionStatePart) (json.RawMessage, error) {
	values := make([]json.RawMessage, 0, len(parts))

	for _, part := range parts {
		switch typed := part.(type) {
		case model.TextPart:
			encoded, err := json.Marshal(typed.Text)
			if err != nil {
				return nil, fmt.Errorf("%w: encode state text: %w", model.ErrInvalidArgument, err)
			}

			values = append(values, encoded)

		case model.DecisionJSONPart:
			values = append(values, typed.Value)

		default:
			return nil, fmt.Errorf("%w: OpenRouter decisions accept text and JSON state, not %T", model.ErrUnsupported, part)
		}
	}

	if len(values) == 1 && isJSONString(values[0]) {
		return values[0], nil
	}

	encoded, err := json.Marshal(values)
	if err != nil {
		return nil, fmt.Errorf("%w: encode state: %w", model.ErrInvalidArgument, err)
	}

	return encoded, nil
}

func isJSONString(value json.RawMessage) bool {
	trimmed := strings.TrimSpace(string(value))

	return strings.HasPrefix(trimmed, `"`)
}

// decisionRequest is a request to the Decisions endpoint, which sits beside
// the "v1" API rather than in it.
func (p *Provider) decisionRequest(callHeaders http.Header) (apicall.Request, error) {
	headers, err := p.requestHeaders()
	if err != nil {
		return apicall.Request{}, err
	}

	return apicall.Request{
		Client:     p.httpClient,
		URL:        strings.TrimSuffix(p.baseURL, "/v1") + decisionPath,
		Headers:    apicall.MergeHeaders(headers, callHeaders),
		ParseError: apicall.ErrorParser(openaicompatible.ParseError),
	}, nil
}

// decisionUsage converts the reported token counts.
func (u *decisionUsageWire) decisionUsage() *model.DecisionUsage {
	if u == nil {
		return &model.DecisionUsage{}
	}

	return &model.DecisionUsage{
		InputTokens:  u.InputTokens,
		OutputTokens: u.OutputTokens,
	}
}

// accounting returns the cost fields, or nil when none were reported.
func (u *decisionUsageWire) accounting() *UsageAccounting {
	if u == nil {
		return nil
	}

	chat := usageWire{
		Cost:        u.Cost,
		IsBYOK:      u.IsBYOK,
		CostDetails: u.CostDetails,
	}

	return chat.accounting()
}
