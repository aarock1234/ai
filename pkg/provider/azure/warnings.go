package azure

import (
	"encoding/json"

	"github.com/aarock1234/ai/pkg/model"
)

// Details of options that only one Azure API supports.
const (
	speechOnlyDetails   = "This option requires the Azure Speech API."
	maiImageOnlyDetails = "This option requires the MAI image API."
	maiStreamingDetails = "This option requires MAI streaming transcription."
)

// unsupportedOptionWarnings reports provider options that the selected API
// ignores.
func unsupportedOptionWarnings(optionNames []string, details string) []model.Warning {
	warnings := make([]model.Warning, 0, len(optionNames))
	for _, name := range optionNames {
		warnings = append(warnings, model.UnsupportedWarning("providerOptions.azure."+name, details))
	}

	return warnings
}

// maiErrorResponse is the error body of the MAI and Azure Speech APIs.
type maiErrorResponse struct {
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`

	// Message is the top-level message of some Azure Speech errors.
	Message string `json:"message"`
}

// parseError reads the message of a failed MAI or Azure Speech response.
func parseError(body []byte) string {
	var response maiErrorResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return ""
	}

	if response.Error != nil {
		return response.Error.Message
	}

	return response.Message
}
