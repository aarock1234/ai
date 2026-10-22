package openai

import "encoding/json"

// errorResponse is the error body returned by OpenAI and compatible APIs.
type errorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

// parseError extracts the message from a failed response body.
func parseError(body []byte) string {
	var response errorResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return ""
	}

	return response.Error.Message
}
