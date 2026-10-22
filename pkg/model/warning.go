package model

import "fmt"

// WarningType classifies a warning.
type WarningType string

// Warning types.
const (
	// WarningUnsupported reports a feature the provider does not support.
	WarningUnsupported WarningType = "unsupported"

	// WarningCompatibility reports a feature that was emulated or mapped
	// approximately.
	WarningCompatibility WarningType = "compatibility"

	// WarningDeprecated reports a deprecated setting.
	WarningDeprecated WarningType = "deprecated"

	// WarningOther reports anything else.
	WarningOther WarningType = "other"
)

// Warning reports a non-fatal problem with a call. Unsupported and
// compatibility warnings use Feature and Details; deprecated warnings use
// Setting and Message; other warnings use Message.
type Warning struct {
	Type    WarningType
	Feature string
	Details string
	Setting string
	Message string
}

// UnsupportedWarning reports that feature is not supported.
func UnsupportedWarning(feature, details string) Warning {
	return Warning{
		Type:    WarningUnsupported,
		Feature: feature,
		Details: details,
	}
}

// CompatibilityWarning reports that feature was applied approximately.
func CompatibilityWarning(feature, details string) Warning {
	return Warning{
		Type:    WarningCompatibility,
		Feature: feature,
		Details: details,
	}
}

// String returns a human-readable description.
func (w Warning) String() string {
	switch w.Type {
	case WarningUnsupported, WarningCompatibility:
		if w.Details == "" {
			return fmt.Sprintf("%s: %s", w.Type, w.Feature)
		}

		return fmt.Sprintf("%s: %s: %s", w.Type, w.Feature, w.Details)

	case WarningDeprecated:
		return fmt.Sprintf("deprecated: %s: %s", w.Setting, w.Message)

	default:
		return w.Message
	}
}
