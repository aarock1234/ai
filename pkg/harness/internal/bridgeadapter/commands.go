package bridgeadapter

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/bridge"
	"github.com/aarock1234/ai/pkg/model"
)

// StartOptions configure NewStartBase.
type StartOptions struct {
	Model          string
	Tools          []harness.ToolSpec
	ResponseFormat *model.ResponseFormat

	PermissionMode       harness.PermissionMode
	BuiltinToolFiltering *harness.BuiltinToolFiltering

	// Observability sends the debug configuration to the bridge when
	// diagnostics are enabled.
	Observability *harness.Observability
}

// NewStartBase returns the shared fields of a start command.
func NewStartBase(options StartOptions) bridge.StartBase {
	base := bridge.StartBase{
		Type:                 "start",
		Tools:                bridge.ToolWiresOf(options.Tools),
		Model:                options.Model,
		PermissionMode:       options.PermissionMode,
		BuiltinToolFiltering: options.BuiltinToolFiltering,
		ResponseFormat:       responseFormatWire(options.ResponseFormat),
	}

	if options.Observability != nil {
		debug := options.Observability.Debug
		base.Debug = &debug
	}

	return base
}

// ValidateResponseFormat rejects JSON response formats without a schema,
// which bridge runtimes cannot enforce.
func ValidateResponseFormat(harnessID string, format *model.ResponseFormat) error {
	if format == nil || format.Type != model.ResponseFormatJSON || len(format.Schema) > 0 {
		return nil
	}

	return &harness.CapabilityUnsupportedError{
		Message:   fmt.Sprintf("Harness '%s' requires a JSON schema for structured output.", harnessID),
		HarnessID: harnessID,
	}
}

// responseFormatWire converts a response format to its wire form.
func responseFormatWire(format *model.ResponseFormat) *bridge.ResponseFormatWire {
	if format == nil {
		return nil
	}

	formatType := format.Type
	if formatType == "" {
		formatType = model.ResponseFormatText
	}

	return &bridge.ResponseFormatWire{
		Type:        string(formatType),
		Schema:      format.Schema,
		Name:        format.Name,
		Description: format.Description,
	}
}

// HeaderMap flattens request headers for a start command, joining repeated
// values with commas. It returns nil for no headers.
func HeaderMap(headers http.Header) map[string]string {
	if len(headers) == 0 {
		return nil
	}

	flattened := make(map[string]string, len(headers))
	for name, values := range headers {
		flattened[name] = strings.Join(values, ", ")
	}

	return flattened
}
