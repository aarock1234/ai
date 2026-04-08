package ai

import "github.com/aarock1234/ai/pkg/model"

// Usage is an alias for model.Usage, representing token consumption.
// In multi-step tool loops, this reflects the totals across all steps.
type Usage = model.Usage
