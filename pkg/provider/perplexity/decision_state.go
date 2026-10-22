package perplexity

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"regexp"
	"slices"

	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/pkg/model"
)

// Limits of the Decisions API.
const (
	maxDecisionQuestions = 128
	maxDecisionChoices   = 255
	maxDecisionLevels    = 10
	maxDecisionImages    = 8
)

// decisionImageMediaTypes are the image types the Decisions API accepts.
var decisionImageMediaTypes = []string{
	"image/png",
	"image/jpeg",
	"image/webp",
}

// decisionImageURLPattern matches the image data URLs the Decisions API
// accepts.
var decisionImageURLPattern = regexp.MustCompile(`(?i)^data:image/(png|jpeg|webp);base64,`)

// validateDecisionQuestion checks the per-question limits.
func validateDecisionQuestion(question model.DecisionQuestion) error {
	if question.QuestionID() == "" {
		return invalidDecisionArgument("Perplexity Decisions requires non-empty question names")
	}

	switch typed := question.(type) {
	case model.ChoiceQuestion:
		if len(typed.Choices) == 0 || len(typed.Choices) > maxDecisionChoices {
			return invalidDecisionArgument("Perplexity Choice questions support 1 to 255 options")
		}

	case model.ScoreQuestion:
		if len(typed.Levels) == 0 || len(typed.Levels) > maxDecisionLevels || slices.Contains(typed.Levels, nil) {
			return invalidDecisionArgument("Perplexity Score questions require 1 to 10 non-null levels")
		}
	}

	return nil
}

// decisionImageURL is an image part of the state.
type decisionImageURL struct {
	Type     string               `json:"type"`
	ImageURL decisionImageAddress `json:"image_url"`
}

type decisionImageAddress struct {
	URL string `json:"url"`
}

// decisionState converts the state. Text becomes a string, JSON stays
// as is and files become image data URLs. A single string, object or array
// is sent alone; other states are sent as an array, because the API rejects
// other values at the top level.
func decisionState(state []model.DecisionStatePart) (json.RawMessage, error) {
	parts := make([]json.RawMessage, 0, len(state))

	for _, part := range state {
		converted, err := decisionStatePart(part)
		if err != nil {
			return nil, err
		}

		parts = append(parts, converted)
	}

	images := 0
	for _, part := range parts {
		count, err := countDecisionImages(part)
		if err != nil {
			return nil, err
		}

		images += count
	}

	if images > maxDecisionImages {
		return nil, invalidDecisionArgument("Perplexity Decisions supports at most 8 images in the shared state")
	}

	if len(parts) == 1 && isStringObjectOrArray(parts[0]) {
		return parts[0], nil
	}

	encoded, err := json.Marshal(parts)
	if err != nil {
		return nil, fmt.Errorf("encode decision state: %w", err)
	}

	return encoded, nil
}

func decisionStatePart(part model.DecisionStatePart) (json.RawMessage, error) {
	switch typed := part.(type) {
	case model.TextPart:
		return json.Marshal(typed.Text)

	case model.DecisionJSONPart:
		return typed.Value, nil

	case model.FilePart:
		return decisionImagePart(typed)

	default:
		return nil, fmt.Errorf("%w: decision state part %T", model.ErrUnsupported, part)
	}
}

// decisionImagePart converts an inline PNG, JPEG or WebP image.
func decisionImagePart(part model.FilePart) (json.RawMessage, error) {
	data, ok := part.Data.(model.FileBytes)
	if !ok {
		return nil, fmt.Errorf("%w: Perplexity decision file input: %T", model.ErrUnsupported, part.Data)
	}

	mediaType := part.MediaType
	if !mediatype.IsFull(mediaType) {
		mediaType, _ = mediatype.Detect(data.Data, "image")
	}

	if !slices.Contains(decisionImageMediaTypes, mediaType) {
		return nil, fmt.Errorf("%w: Perplexity decision image media type: %s", model.ErrUnsupported, part.MediaType)
	}

	return json.Marshal(decisionImageURL{
		Type: "image_url",
		ImageURL: decisionImageAddress{
			URL: dataurl.Encode(mediaType, data.Data),
		},
	})
}

// imageURLHeader identifies image parts nested anywhere in the state.
type imageURLHeader struct {
	Type     string          `json:"type"`
	ImageURL json.RawMessage `json:"image_url"`
}

// countDecisionImages counts the image parts in a state value, including
// those nested in JSON, and checks that each is an accepted data URL.
func countDecisionImages(value json.RawMessage) (int, error) {
	trimmed := bytes.TrimSpace(value)
	if len(trimmed) == 0 {
		return 0, nil
	}

	switch trimmed[0] {
	case '[':
		var items []json.RawMessage
		if err := json.Unmarshal(trimmed, &items); err != nil {
			return 0, fmt.Errorf("%w: decision state: %w", model.ErrInvalidArgument, err)
		}

		return countNestedImages(items)

	case '{':
		return countObjectImages(trimmed)

	default:
		return 0, nil
	}
}

func countObjectImages(object json.RawMessage) (int, error) {
	var header imageURLHeader
	if err := json.Unmarshal(object, &header); err != nil {
		return 0, fmt.Errorf("%w: decision state: %w", model.ErrInvalidArgument, err)
	}

	if header.Type == "image_url" {
		if !isAcceptedImageURL(header.ImageURL) {
			return 0, fmt.Errorf("%w: Perplexity decision images require PNG, JPEG, or WebP base64 data URLs", model.ErrUnsupported)
		}

		return 1, nil
	}

	var members map[string]json.RawMessage
	if err := json.Unmarshal(object, &members); err != nil {
		return 0, fmt.Errorf("%w: decision state: %w", model.ErrInvalidArgument, err)
	}

	return countNestedImages(slices.Collect(maps.Values(members)))
}

func countNestedImages(values []json.RawMessage) (int, error) {
	total := 0

	for _, value := range values {
		count, err := countDecisionImages(value)
		if err != nil {
			return 0, err
		}

		total += count
	}

	return total, nil
}

func isAcceptedImageURL(raw json.RawMessage) bool {
	var address decisionImageAddress
	if json.Unmarshal(raw, &address) != nil {
		return false
	}

	return decisionImageURLPattern.MatchString(address.URL)
}

func isStringObjectOrArray(value json.RawMessage) bool {
	trimmed := bytes.TrimSpace(value)

	return len(trimmed) > 0 && (trimmed[0] == '"' || trimmed[0] == '{' || trimmed[0] == '[')
}

func invalidDecisionArgument(message string) error {
	return fmt.Errorf("%w: %s", model.ErrInvalidArgument, message)
}
