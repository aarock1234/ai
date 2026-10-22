package typesafeai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

// FilePartOptions are the image settings read from the "typesafe" provider
// options of a file part.
type FilePartOptions struct {
	// Reference names the image in the state, instructions or criteria. It
	// defaults to a generated ID. References are nonempty, unique within
	// the request, and should not occur accidentally in other text. Every
	// occurrence sends another copy of the image.
	Reference *string `json:"reference,omitempty"`

	// InsertReference sets whether the reference is inserted into the
	// state at the file's position. It defaults to inserting the reference
	// unless the state already contains it. False requires an explicit
	// Reference.
	InsertReference *bool `json:"insertReference,omitempty"`
}

// imageAsset maps a reference to the form part that holds the image.
type imageAsset struct {
	Type string `json:"type"`
	Part string `json:"part"`
}

// imagePart is an image sent as a form part.
type imagePart struct {
	name      string
	filename  string
	mediaType string
	data      []byte
}

// preparedState is the state of a request with its image assets.
type preparedState struct {
	state  json.RawMessage
	assets map[string]imageAsset
	images []imagePart
}

// statePart is a text or JSON part of the state after images were replaced
// by their references.
type statePart struct {
	text  *string
	value json.RawMessage
}

// prepareDecisionState replaces images by their references and encodes the
// state: a single JSON part is sent as its value, and other states as their
// parts joined by newlines.
func prepareDecisionState(state []model.DecisionStatePart, generateID func() string) (*preparedState, error) {
	prepared := &preparedState{
		assets: make(map[string]imageAsset),
	}

	stateStrings, err := decisionStateStrings(state)
	if err != nil {
		return nil, err
	}

	var parts []statePart

	for index, part := range state {
		converted, keep, err := prepared.convertPart(index, part, stateStrings, generateID)
		if err != nil {
			return nil, err
		}

		if keep {
			parts = append(parts, converted)
		}
	}

	encoded, err := encodeState(parts)
	if err != nil {
		return nil, err
	}

	prepared.state = encoded

	return prepared, nil
}

func (p *preparedState) convertPart(index int, part model.DecisionStatePart, stateStrings []string, generateID func() string) (statePart, bool, error) {
	switch typed := part.(type) {
	case model.TextPart:
		return statePart{
			text: new(typed.Text),
		}, true, nil

	case model.DecisionJSONPart:
		return statePart{
			value: typed.Value,
		}, true, nil

	case model.FilePart:
		reference, insert, err := p.addImage(index, typed, stateStrings, generateID)
		if err != nil || !insert {
			return statePart{}, false, err
		}

		return statePart{
			text: new(reference),
		}, true, nil

	default:
		return statePart{}, false, fmt.Errorf("%w: decision state part %T", model.ErrUnsupported, part)
	}
}

// addImage registers an inline PNG or JPEG image as an asset. It returns
// the image's reference and whether to insert it into the state.
func (p *preparedState) addImage(index int, part model.FilePart, stateStrings []string, generateID func() string) (string, bool, error) {
	data, ok := part.Data.(model.FileBytes)
	if !ok {
		return "", false, fmt.Errorf("%w: TypeSafe AI decision file input: %s (%T)", model.ErrUnsupported, part.MediaType, part.Data)
	}

	if part.MediaType != "image/png" && part.MediaType != "image/jpeg" {
		return "", false, fmt.Errorf("%w: TypeSafe AI decision image media type: %s", model.ErrUnsupported, part.MediaType)
	}

	options, err := readFilePartOptions(part.ProviderOptions)
	if err != nil {
		return "", false, err
	}

	reference := generateID()
	if options.Reference != nil {
		reference = *options.Reference
	}

	insert := !slices.ContainsFunc(stateStrings, func(text string) bool {
		return strings.Contains(text, reference)
	})

	if options.InsertReference != nil {
		insert = *options.InsertReference
	}

	if _, duplicate := p.assets[reference]; duplicate {
		return "", false, fmt.Errorf("%w: providerOptions.typesafe.reference: duplicate TypeSafe image reference: %s", model.ErrInvalidArgument, reference)
	}

	name := "image-" + strconv.Itoa(index)

	p.assets[reference] = imageAsset{
		Type: "image",
		Part: name,
	}

	p.images = append(p.images, imagePart{
		name:      name,
		filename:  imageFilename(part, name),
		mediaType: part.MediaType,
		data:      data.Data,
	})

	return reference, insert, nil
}

func readFilePartOptions(providerOptions model.ProviderOptions) (FilePartOptions, error) {
	options, _, err := model.ProviderValue[FilePartOptions](providerOptions, Namespace)
	if err != nil {
		return FilePartOptions{}, err
	}

	if options.Reference != nil && strings.TrimSpace(*options.Reference) == "" {
		return FilePartOptions{}, fmt.Errorf("%w: providerOptions.typesafe.reference must not be blank", model.ErrInvalidArgument)
	}

	if options.InsertReference != nil && !*options.InsertReference && options.Reference == nil {
		return FilePartOptions{}, fmt.Errorf(
			"%w: providerOptions.typesafe.reference: TypeSafe images with insertReference: false need an explicit reference",
			model.ErrInvalidArgument,
		)
	}

	return options, nil
}

func imageFilename(part model.FilePart, name string) string {
	if part.Filename != "" {
		return part.Filename
	}

	if part.MediaType == "image/png" {
		return name + ".png"
	}

	return name + ".jpeg"
}

// encodeState returns a single JSON part as its value, and otherwise the
// parts joined by newlines, with JSON parts as compact JSON.
func encodeState(parts []statePart) (json.RawMessage, error) {
	if len(parts) == 1 && parts[0].text == nil {
		return parts[0].value, nil
	}

	lines := make([]string, 0, len(parts))

	for _, part := range parts {
		if part.text != nil {
			lines = append(lines, *part.text)

			continue
		}

		lines = append(lines, model.DecisionInputText(model.DecisionJSON(part.value)))
	}

	return json.Marshal(strings.Join(lines, "\n"))
}

// decisionStateStrings returns the text of the state: text parts and every
// string inside JSON parts.
func decisionStateStrings(state []model.DecisionStatePart) ([]string, error) {
	var texts []string

	for _, part := range state {
		switch typed := part.(type) {
		case model.TextPart:
			texts = append(texts, typed.Text)

		case model.DecisionJSONPart:
			nested, err := jsonStrings(typed.Value)
			if err != nil {
				return nil, err
			}

			texts = append(texts, nested...)
		}
	}

	return texts, nil
}

// jsonStrings returns every string in a JSON value.
func jsonStrings(value json.RawMessage) ([]string, error) {
	trimmed := bytes.TrimSpace(value)
	if len(trimmed) == 0 {
		return nil, nil
	}

	switch trimmed[0] {
	case '"':
		var text string
		if err := json.Unmarshal(trimmed, &text); err != nil {
			return nil, invalidState(err)
		}

		return []string{
			text,
		}, nil

	case '[':
		var items []json.RawMessage
		if err := json.Unmarshal(trimmed, &items); err != nil {
			return nil, invalidState(err)
		}

		return nestedStrings(items)

	case '{':
		var members map[string]json.RawMessage
		if err := json.Unmarshal(trimmed, &members); err != nil {
			return nil, invalidState(err)
		}

		values := make([]json.RawMessage, 0, len(members))
		for _, member := range members {
			values = append(values, member)
		}

		return nestedStrings(values)

	default:
		return nil, nil
	}
}

func nestedStrings(values []json.RawMessage) ([]string, error) {
	var texts []string

	for _, value := range values {
		nested, err := jsonStrings(value)
		if err != nil {
			return nil, err
		}

		texts = append(texts, nested...)
	}

	return texts, nil
}

func invalidState(err error) error {
	return fmt.Errorf("%w: decision state: %w", model.ErrInvalidArgument, err)
}
