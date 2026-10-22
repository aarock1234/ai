package azure

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// voiceResetReason marks the 502 that Azure Speech returns for unknown voices
// or styles.
const voiceResetReason = "reset reason: protocol error"

// send posts the SSML document and returns the audio response.
func (m *speechSynthesisModel) send(ctx context.Context, ssml string, outputFormat string, options model.SpeechOptions) (*apicall.Response, error) {
	requestURL, err := m.settings.speechURL(speechSynthesisPath)
	if err != nil {
		return nil, err
	}

	formatHeader := http.Header{
		"X-Microsoft-Outputformat": {
			outputFormat,
		},
	}

	request, err := m.settings.serviceRequest(
		apiSpeech,
		requestURL,
		apicall.MergeHeaders(formatHeader, options.Headers),
	)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostBody(ctx, request, []byte(ssml), "application/ssml+xml")
	if err != nil {
		return nil, synthesisError(err)
	}

	return response, nil
}

// synthesisError describes a failed synthesis. Azure Speech answers invalid
// styles or output formats with an empty 400 body, and resets the connection
// for unknown voices or styles with a 502 whose reason is "protocol error".
// That reset is a client input error, so it is reported as a non-retryable
// 400; other 502s stay retryable. Errors without a response pass through.
func synthesisError(err error) error {
	var callError *model.APICallError
	if !errors.As(err, &callError) || callError.StatusCode == 0 {
		return err
	}

	isVoiceReset := callError.StatusCode == http.StatusBadGateway &&
		strings.Contains(callError.ResponseBody, voiceResetReason)

	adapted := *callError
	adapted.Message = synthesisErrorMessage([]byte(callError.ResponseBody), callError.StatusCode, isVoiceReset)

	if isVoiceReset {
		adapted.StatusCode = http.StatusBadRequest
		adapted.IsRetryable = false
	}

	return &adapted
}

// speechErrorBody is the JSON error body of Azure Speech.
type speechErrorBody struct {
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// parseSpeechError returns the message of an Azure Speech JSON error body.
func parseSpeechError(body []byte) string {
	var decoded speechErrorBody
	if err := json.Unmarshal(body, &decoded); err != nil || decoded.Error == nil {
		return ""
	}

	return decoded.Error.Message
}

func synthesisErrorMessage(body []byte, status int, isVoiceReset bool) string {
	if message := parseSpeechError(body); message != "" {
		return message
	}

	switch {
	case isVoiceReset:
		return "Azure Speech could not synthesize the request. Check that the voice is available for this model and that the style is supported by the voice."
	case status == http.StatusBadRequest:
		return "Azure Speech request failed with status 400. Check the voice name, style, and output format."
	default:
		return fmt.Sprintf("Azure Speech request failed with status %d.", status)
	}
}
