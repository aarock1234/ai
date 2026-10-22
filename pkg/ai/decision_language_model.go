package ai

import (
	"github.com/aarock1234/ai/internal/decisionlm"
	"github.com/aarock1234/ai/pkg/model"
)

// NewDecisionLanguageModel adapts a language model with structured output
// into a decision model for Decide. It answers choice, score and boolean
// questions with one call whose JSON schema has one property per question.
// An empty provider reports the language model's provider with a
// ".decision" suffix. It ports experimental DecisionLanguageModel and is
// experimental.
func NewDecisionLanguageModel(languageModel model.LanguageModel, provider string) model.DecisionModel {
	return decisionlm.New(languageModel, provider)
}
