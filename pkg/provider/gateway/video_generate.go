package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/sse"
	"github.com/aarock1234/ai/pkg/model"
)

// Event types of the generation stream.
const (
	videoEventResult = "result"
	videoEventError  = "error"
)

// videoResultEvent is the result event of the generation stream.
type videoResultEvent struct {
	Videos           []wireVideo   `json:"videos"`
	Warnings         []wireWarning `json:"warnings"`
	ProviderMetadata wireMetadata  `json:"providerMetadata"`
}

// videoErrorEvent is the error event of the generation stream.
type videoErrorEvent struct {
	Message    string          `json:"message"`
	ErrorType  string          `json:"errorType"`
	StatusCode int             `json:"statusCode"`
	Param      json.RawMessage `json:"param"`
}

// videoErrorData is the documented error body that an error event becomes.
type videoErrorData struct {
	Error videoErrorDetail `json:"error"`
}

type videoErrorDetail struct {
	Message string          `json:"message"`
	Type    string          `json:"type"`
	Param   json.RawMessage `json:"param,omitempty"`
}

// Generate generates videos in one call. The gateway keeps the connection
// open with an event stream and sends one result or error event.
func (m *VideoModel) Generate(ctx context.Context, options model.VideoOptions) (*model.VideoResult, error) {
	body, err := videoBody(options)
	if err != nil {
		return nil, err
	}

	modelHeaders := apicall.MergeHeaders(modelConfigHeaders("video", m.modelID), http.Header{
		"Accept": {
			"text/event-stream",
		},
	})

	request, method, err := m.provider.modelRequest(videoModelPath, options.Headers, modelHeaders)
	if err != nil {
		return nil, err
	}

	stream, err := apicall.PostJSONStream(ctx, request, body)
	if err != nil {
		return nil, asGatewayError(err, method)
	}
	defer closeVideoStream(stream)

	event, err := firstVideoEvent(ctx, stream, request.URL)
	if err != nil {
		return nil, asGatewayError(err, method)
	}

	return m.videoResult(event, stream, request.URL, method)
}

// firstVideoEvent reads the first data event of the stream.
func firstVideoEvent(ctx context.Context, stream *apicall.Stream, url string) (json.RawMessage, error) {
	for event, err := range sse.Read(stream.Body) {
		if err != nil {
			return nil, readError(ctx, err)
		}

		if event.Data == "" || event.Data == doneMarker {
			continue
		}

		return json.RawMessage(event.Data), nil
	}

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	return nil, streamAPIError(stream, url, "SSE stream ended without a data event", nil)
}

// videoResult converts the result event, or reports the error event as a
// gateway error.
func (m *VideoModel) videoResult(
	event json.RawMessage,
	stream *apicall.Stream,
	url string,
	method authMethod,
) (*model.VideoResult, error) {
	header, err := decodeAs[typeHeader](event)
	if err != nil {
		return nil, asGatewayError(streamAPIError(stream, url, "Failed to parse video SSE event", err), method)
	}

	switch header.Type {
	case videoEventResult:
		result, err := m.decodeResultEvent(event, stream)
		if err != nil {
			return nil, asGatewayError(streamAPIError(stream, url, "Failed to parse video SSE event", err), method)
		}

		return result, nil

	case videoEventError:
		errorEvent, err := decodeAs[videoErrorEvent](event)
		if err != nil {
			return nil, asGatewayError(streamAPIError(stream, url, "Failed to parse video SSE event", err), method)
		}

		return nil, asGatewayError(errorEventAPIError(errorEvent, stream, url), method)

	default:
		parseErr := fmt.Errorf("%w: unknown video event type %q", model.ErrInvalidResponse, header.Type)

		return nil, asGatewayError(streamAPIError(stream, url, "Failed to parse video SSE event", parseErr), method)
	}
}

func (m *VideoModel) decodeResultEvent(event json.RawMessage, stream *apicall.Stream) (*model.VideoResult, error) {
	result, err := decodeAs[videoResultEvent](event)
	if err != nil {
		return nil, err
	}

	videos, err := decodeVideos(result.Videos)
	if err != nil {
		return nil, err
	}

	return &model.VideoResult{
		Videos:           videos,
		ProviderMetadata: result.ProviderMetadata.providerMetadata(),
		Warnings:         decodeWarnings(result.Warnings),
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: time.Now(),
				ModelID:   m.modelID,
			},
			Headers: stream.Headers,
			Body:    event,
		},
	}, nil
}

// errorEventAPIError describes an error event as a failed call whose body is
// the documented error format, so that it becomes a typed gateway error.
func errorEventAPIError(event videoErrorEvent, stream *apicall.Stream, url string) error {
	body, err := json.Marshal(videoErrorData{
		Error: videoErrorDetail{
			Message: event.Message,
			Type:    event.ErrorType,
			Param:   event.Param,
		},
	})
	if err != nil {
		return fmt.Errorf("encode video error event: %w", err)
	}

	return &model.APICallError{
		Message:         event.Message,
		URL:             url,
		StatusCode:      event.StatusCode,
		ResponseHeaders: stream.Headers,
		ResponseBody:    string(body),
		IsRetryable:     model.RetryableStatus(event.StatusCode),
	}
}

// streamAPIError reports a stream that did not deliver a usable event.
func streamAPIError(stream *apicall.Stream, url string, message string, cause error) error {
	return &model.APICallError{
		Message:         message,
		URL:             url,
		StatusCode:      http.StatusOK,
		ResponseHeaders: stream.Headers,
		Cause:           cause,
	}
}

// closeVideoStream closes the event stream after its first event. The close
// error is ignored: the outcome of the call is already decided by then.
func closeVideoStream(stream *apicall.Stream) {
	_ = stream.Body.Close()
}
