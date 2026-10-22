package openaicompatible

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/pkg/model"
)

// formFile is an image sent in a multipart form.
type formFile struct {
	MediaType string
	Data      []byte
}

// imageForm collects multipart fields in order. Setting a name again replaces
// its earlier values in place, like assigning an object key.
type imageForm struct {
	names  []string
	values map[string][]apicall.FormField
}

func (f *imageForm) set(name string, fields []apicall.FormField) {
	if f.values == nil {
		f.values = make(map[string][]apicall.FormField)
	}

	if _, ok := f.values[name]; !ok {
		f.names = append(f.names, name)
	}

	f.values[name] = fields
}

func (f *imageForm) setValue(name string, value string) {
	f.set(name, []apicall.FormField{
		{
			Name:  name,
			Value: value,
		},
	})
}

// setFiles adds the files under name, or under "name[]" when there are
// several.
func (f *imageForm) setFiles(name string, files []formFile) {
	fieldName := arrayFieldName(name, len(files))

	fields := make([]apicall.FormField, 0, len(files))
	for _, file := range files {
		fields = append(fields, apicall.FormField{
			Name:      fieldName,
			Filename:  name + fileExtension(file.MediaType),
			MediaType: file.MediaType,
			Data:      file.Data,
		})
	}

	f.set(name, fields)
}

// setJSONMembers adds provider options. Strings are sent unquoted, arrays as
// repeated fields, null values are skipped and other values as JSON text.
func (f *imageForm) setJSONMembers(members map[string]json.RawMessage) error {
	names := slices.Sorted(maps.Keys(members))

	for _, name := range names {
		value := members[name]
		if isJSONNull(value) {
			continue
		}

		var items []json.RawMessage
		if err := json.Unmarshal(value, &items); err != nil {
			items = []json.RawMessage{
				value,
			}
		}

		fieldName := arrayFieldName(name, len(items))

		fields := make([]apicall.FormField, 0, len(items))
		for _, item := range items {
			text, err := formText(item)
			if err != nil {
				return fmt.Errorf("%w: provider option %q: %w", model.ErrInvalidArgument, name, err)
			}

			fields = append(fields, apicall.FormField{
				Name:  fieldName,
				Value: text,
			})
		}

		f.set(name, fields)
	}

	return nil
}

func (f *imageForm) fields() []apicall.FormField {
	var fields []apicall.FormField
	for _, name := range f.names {
		fields = append(fields, f.values[name]...)
	}

	return fields
}

// arrayFieldName uses bracket notation for repeated fields.
func arrayFieldName(name string, count int) string {
	if count > 1 {
		return name + "[]"
	}

	return name
}

// formText returns a JSON string's value and other JSON values as text.
func formText(value json.RawMessage) (string, error) {
	trimmed := bytes.TrimSpace(value)
	if !bytes.HasPrefix(trimmed, []byte(`"`)) {
		return string(trimmed), nil
	}

	var text string
	if err := json.Unmarshal(trimmed, &text); err != nil {
		return "", err
	}

	return text, nil
}

func isJSONNull(value json.RawMessage) bool {
	return bytes.Equal(bytes.TrimSpace(value), []byte("null"))
}

// fileExtension returns ".png" for "image/png", or nothing when the media
// type has no subtype.
func fileExtension(mediaType string) string {
	if !mediatype.IsFull(mediaType) {
		return ""
	}

	_, subtype, _ := strings.Cut(mediaType, "/")

	return "." + subtype
}
