package google

import (
	"context"
	"net/http"

	"github.com/aarock1234/ai/internal/livetranscription"
)

// liveWebSocketPath is the Live API service of the Gemini API.
const liveWebSocketPath = "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"

// liveDialOptions describes the Live API WebSocket of a call. The API key
// moves from its header to the key query parameter, which the Live API
// requires, and errors name the URL without it. purpose names the call when
// the key is missing, such as "streaming transcription".
func (p *Provider) liveDialOptions(ctx context.Context, callHeaders http.Header, purpose string) (livetranscription.DialOptions, error) {
	request, err := p.request(ctx, "", callHeaders)
	if err != nil {
		return livetranscription.DialOptions{}, err
	}

	apiKey, err := liveAPIKeyOf(request.Headers, purpose)
	if err != nil {
		return livetranscription.DialOptions{}, err
	}

	baseURL, err := p.currentBaseURL()
	if err != nil {
		return livetranscription.DialOptions{}, err
	}

	target, err := realtimeWebSocketURL(baseURL, liveWebSocketPath)
	if err != nil {
		return livetranscription.DialOptions{}, err
	}

	displayURL := target.String()

	query := target.Query()
	query.Set("key", apiKey)
	target.RawQuery = query.Encode()

	headers := request.Headers.Clone()
	headers.Del(apiKeyHeader)

	return livetranscription.DialOptions{
		URL:        target.String(),
		DisplayURL: displayURL,
		Client:     request.Client,
		Header:     headers,
		ParseError: request.ParseError,
	}, nil
}
