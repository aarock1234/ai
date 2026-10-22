package anthropic

import (
	"context"

	"github.com/aarock1234/ai/internal/decisionlm"
	"github.com/aarock1234/ai/pkg/model"
)

var (
	_ model.DecisionProvider = (*Provider)(nil)
	_ model.DecisionModel    = (*DecisionModel)(nil)
)

// DecisionModel answers choice, score and boolean questions with one
// structured-output Messages API call. Decision models are experimental.
type DecisionModel struct {
	decider *decisionlm.Model
}

// Decision returns a decision model. Its provider is "anthropic.decision"
// by default.
func (p *Provider) Decision(modelID string) *DecisionModel {
	return &DecisionModel{
		decider: decisionlm.New(p.Messages(modelID), p.baseName()+".decision"),
	}
}

// DecisionModel returns Decision(modelID).
func (p *Provider) DecisionModel(modelID string) (model.DecisionModel, error) {
	return p.Decision(modelID), nil
}

// Provider returns the provider identifier, such as "anthropic.decision".
func (m *DecisionModel) Provider() string {
	return m.decider.Provider()
}

// ModelID returns the model ID.
func (m *DecisionModel) ModelID() string {
	return m.decider.ModelID()
}

// SupportedQuestionTypes returns choice, score and boolean.
func (m *DecisionModel) SupportedQuestionTypes() []model.DecisionQuestionType {
	return m.decider.SupportedQuestionTypes()
}

// Decide answers every question against the same state.
func (m *DecisionModel) Decide(ctx context.Context, options model.DecisionOptions) (*model.DecisionResult, error) {
	return m.decider.Decide(ctx, options)
}
