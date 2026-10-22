package ai

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/aarock1234/ai/pkg/model"
)

// LogWarningsFunc receives the warnings of a model call, for example to log
// them. Upstream logs warnings to the console by default and reads a global
// override; Go has no global logger, so warnings are only logged when a call
// sets a LogWarningsFunc. They are always returned on results.
type LogWarningsFunc func(ctx context.Context, event WarningsEvent)

// WarningsEvent carries the warnings of one call and the model they came
// from.
type WarningsEvent struct {
	Warnings []model.Warning
	Provider string
	ModelID  string
}

// SlogWarnings returns a LogWarningsFunc that logs each warning with logger
// at the warn level.
func SlogWarnings(logger *slog.Logger) LogWarningsFunc {
	return func(ctx context.Context, event WarningsEvent) {
		for _, warning := range event.Warnings {
			logger.WarnContext(ctx, "model call warning",
				slog.String("provider", event.Provider),
				slog.String("model", event.ModelID),
				slog.String("type", string(warning.Type)),
				slog.String("warning", formatWarning(warning)),
			)
		}
	}
}

// log passes non-empty warnings to the function. A nil function discards
// them.
func (f LogWarningsFunc) log(ctx context.Context, event WarningsEvent) {
	if f == nil || len(event.Warnings) == 0 {
		return
	}

	f(ctx, event)
}

// formatWarning describes a warning in upstream's wording, without its
// console prefix.
func formatWarning(warning model.Warning) string {
	switch warning.Type {
	case model.WarningUnsupported:
		return withDetails(fmt.Sprintf("The feature %q is not supported.", warning.Feature), warning.Details)

	case model.WarningCompatibility:
		return withDetails(fmt.Sprintf("The feature %q is used in a compatibility mode.", warning.Feature), warning.Details)

	case model.WarningDeprecated:
		return fmt.Sprintf("Deprecated: %q. %s", warning.Setting, warning.Message)

	case model.WarningOther:
		return warning.Message

	default:
		return warning.String()
	}
}

func withDetails(message, details string) string {
	if details == "" {
		return message
	}

	return message + " " + details
}
