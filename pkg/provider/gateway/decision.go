package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/decisionwire"
	"github.com/aarock1234/ai/pkg/model"
)

// decisionModelPath is the decision model endpoint.
const decisionModelPath = "/decision-model"

var (
	_ model.DecisionModel    = (*DecisionModel)(nil)
	_ model.DecisionProvider = (*Provider)(nil)
)

// Decision returns a decision model, such as "openai/gpt-6-luna-decisions"
// or "perplexity/pplx-decider-v1.2". Decision models are experimental.
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

// DecisionModel is an AI Gateway decision model. It forwards the state and
// the questions in the normalized form and returns the gateway's normalized
// answers. The decision contract is experimental and may change.
type DecisionModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "gateway" by default.
func (m *DecisionModel) Provider() string {
	return m.provider.name
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

// decisionRequest is the body of a decision call.
type decisionRequest struct {
	StateParts      []json.RawMessage      `json:"stateParts"`
	Questions       decisionwire.Questions `json:"questions"`
	MaxOutputTokens *int                   `json:"maxOutputTokens,omitempty"`
	Reasoning       model.ReasoningEffort  `json:"reasoning,omitempty"`
	ProviderOptions model.ProviderOptions  `json:"providerOptions,omitempty"`
}

// decisionJSONPart is a JSON state part on the wire.
type decisionJSONPart struct {
	Type  string          `json:"type"`
	Value json.RawMessage `json:"value"`
}

// decisionResponse is the normalized decision result on the wire.
type decisionResponse struct {
	Answers          map[string]decisionwire.Answer `json:"answers"`
	Model            string                         `json:"model"`
	Rounding         *decisionRounding              `json:"rounding"`
	Usage            *decisionUsage                 `json:"usage"`
	Warnings         []wireWarning                  `json:"warnings"`
	ProviderMetadata wireMetadata                   `json:"providerMetadata"`
}

type decisionRounding struct {
	ProbabilityDecimals *int `json:"probabilityDecimals"`
	ScoreDecimals       *int `json:"scoreDecimals"`
}

type decisionUsage struct {
	InputTokens  *int `json:"inputTokens"`
	OutputTokens *int `json:"outputTokens"`
}

// Decide answers the questions with one gateway call.
func (m *DecisionModel) Decide(ctx context.Context, options model.DecisionOptions) (*model.DecisionResult, error) {
	body, err := m.decisionRequest(options)
	if err != nil {
		return nil, err
	}

	request, method, err := m.provider.modelRequest(decisionModelPath, options.Headers, m.configHeaders())
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, asGatewayError(err, method)
	}

	var decoded decisionResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	answers, err := decisionwire.ConvertAnswers(decoded.Answers, options.Questions, "AI Gateway decisions")
	if err != nil {
		return nil, err
	}

	modelID := decoded.Model
	if modelID == "" {
		modelID = m.modelID
	}

	return &model.DecisionResult{
		Answers:          answers,
		Rounding:         decoded.Rounding.rounding(),
		Usage:            decoded.Usage.usage(),
		Warnings:         decodeWarnings(decoded.Warnings),
		ProviderMetadata: decoded.ProviderMetadata.providerMetadata(),
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				ModelID: modelID,
			},
			Headers: response.Headers,
			Body:    response.Body,
		},
	}, nil
}

func (m *DecisionModel) decisionRequest(options model.DecisionOptions) (decisionRequest, error) {
	stateParts, err := encodeDecisionState(options.State)
	if err != nil {
		return decisionRequest{}, err
	}

	questions := make(decisionwire.Questions, 0, len(options.Questions))

	for _, question := range options.Questions {
		encoded, supported, err := decisionwire.EncodeQuestion(question, decisionwire.NormalizedQuestionTypes)
		if err != nil {
			return decisionRequest{}, err
		}

		if !supported {
			return decisionRequest{}, m.unsupportedQuestion(question)
		}

		questions = append(questions, decisionwire.NamedQuestion{
			ID:       question.QuestionID(),
			Question: encoded,
		})
	}

	return decisionRequest{
		StateParts:      stateParts,
		Questions:       questions,
		MaxOutputTokens: options.MaxOutputTokens,
		Reasoning:       options.Reasoning,
		ProviderOptions: options.ProviderOptions,
	}, nil
}

func (m *DecisionModel) unsupportedQuestion(question model.DecisionQuestion) error {
	return &model.DecisionUnsupportedQuestionTypeError{
		QuestionID:   question.QuestionID(),
		QuestionType: question.QuestionType(),
		Provider:     m.Provider(),
		ModelID:      m.modelID,
	}
}

// encodeDecisionState converts the state parts. Files must be inline bytes,
// which travel as base64.
func encodeDecisionState(state []model.DecisionStatePart) ([]json.RawMessage, error) {
	encoded := make([]json.RawMessage, 0, len(state))

	for _, part := range state {
		value, err := encodeDecisionStatePart(part)
		if err != nil {
			return nil, err
		}

		raw, err := json.Marshal(value)
		if err != nil {
			return nil, fmt.Errorf("encode decision state: %w", err)
		}

		encoded = append(encoded, raw)
	}

	return encoded, nil
}

func encodeDecisionStatePart(part model.DecisionStatePart) (any, error) {
	switch typed := part.(type) {
	case model.TextPart:
		return encodeUserPart(typed)

	case model.FilePart:
		if _, inline := typed.Data.(model.FileBytes); !inline {
			return nil, fmt.Errorf("%w: Gateway decision file input: %T data", model.ErrUnsupported, typed.Data)
		}

		return encodeUserPart(typed)

	case model.DecisionJSONPart:
		return decisionJSONPart{
			Type:  "json",
			Value: typed.Value,
		}, nil

	default:
		return nil, fmt.Errorf("%w: decision state part %T", model.ErrUnsupported, part)
	}
}

// configHeaders describe the call to the gateway.
func (m *DecisionModel) configHeaders() http.Header {
	return http.Header{
		"Ai-Decision-Model-Specification-Version": {
			"4",
		},
		"Ai-Model-Id": {
			m.modelID,
		},
	}
}

func (r *decisionRounding) rounding() *model.DecisionRounding {
	if r == nil {
		return nil
	}

	return &model.DecisionRounding{
		ProbabilityDecimals: r.ProbabilityDecimals,
		ScoreDecimals:       r.ScoreDecimals,
	}
}

func (u *decisionUsage) usage() *model.DecisionUsage {
	if u == nil {
		return nil
	}

	return &model.DecisionUsage{
		InputTokens:  u.InputTokens,
		OutputTokens: u.OutputTokens,
	}
}
