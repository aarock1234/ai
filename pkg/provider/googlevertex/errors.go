package googlevertex

import "encoding/json"

// errorResponse is the error body of Google Cloud APIs.
type errorResponse struct {
	Error *struct {
		Code    *int   `json:"code"`
		Message string `json:"message"`
		Status  string `json:"status"`
	} `json:"error"`
}

// parseError reads the message of a failed response.
func parseError(body []byte) string {
	var response errorResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return ""
	}

	if response.Error == nil {
		return ""
	}

	return response.Error.Message
}
