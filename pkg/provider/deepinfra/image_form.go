package deepinfra

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/download"
	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// blobFilename is the filename that form uploads of unnamed files receive.
const blobFilename = "blob"

// formEntry is the fields of one form member. A member with several values
// repeats its name, without brackets.
type formEntry struct {
	name   string
	fields []apicall.FormField
}

// editForm collects form members in order. Setting a member again replaces
// its fields in place, like assigning an object key.
type editForm struct {
	entries []formEntry
}

func (f *editForm) set(name string, fields []apicall.FormField) {
	for index, entry := range f.entries {
		if entry.name == name {
			f.entries[index].fields = fields

			return
		}
	}

	f.entries = append(f.entries, formEntry{
		name:   name,
		fields: fields,
	})
}

func (f *editForm) setValue(name string, value string) {
	f.set(name, []apicall.FormField{
		{
			Name:  name,
			Value: value,
		},
	})
}

func (f *editForm) fields() []apicall.FormField {
	var fields []apicall.FormField
	for _, entry := range f.entries {
		fields = append(fields, entry.fields...)
	}

	return fields
}

// editFormFields returns the members of an edit request: the model, prompt,
// images, mask, count and size, followed by the provider options.
func editFormFields(
	ctx context.Context,
	client *http.Client,
	modelID string,
	options model.ImageOptions,
	imageOptions jsonobject.Object,
) ([]apicall.FormField, error) {
	var form editForm
	form.setValue("model", modelID)

	if options.Prompt != "" {
		form.setValue("prompt", options.Prompt)
	}

	images := make([]apicall.FormField, 0, len(options.Files))
	for _, file := range options.Files {
		image, err := imageField(ctx, client, "image", file)
		if err != nil {
			return nil, err
		}

		images = append(images, image)
	}

	form.set("image", images)

	if options.Mask != nil {
		mask, err := imageField(ctx, client, "mask", options.Mask)
		if err != nil {
			return nil, err
		}

		form.set("mask", []apicall.FormField{
			mask,
		})
	}

	if options.N > 0 {
		form.setValue("n", strconv.Itoa(options.N))
	}

	if options.Size != "" {
		form.setValue("size", options.Size)
	}

	for _, member := range imageOptions {
		fields, err := optionFields(member)
		if err != nil {
			return nil, err
		}

		form.set(member.Key, fields)
	}

	return form.fields(), nil
}

// imageField returns an input image as a file field. URL images are
// downloaded first.
func imageField(ctx context.Context, client *http.Client, name string, file model.ImageFile) (apicall.FormField, error) {
	switch file := file.(type) {
	case model.ImageBytes:
		return apicall.FormField{
			Name:      name,
			Filename:  blobFilename,
			MediaType: file.MediaType,
			Data:      file.Data,
		}, nil

	case model.ImageURL:
		downloaded, err := download.Fetch(ctx, download.Request{
			URL:    file.URL,
			Client: client,
		})
		if err != nil {
			return apicall.FormField{}, err
		}

		return apicall.FormField{
			Name:      name,
			Filename:  blobFilename,
			MediaType: downloaded.MediaType,
			Data:      downloaded.Data,
		}, nil

	default:
		return apicall.FormField{}, fmt.Errorf("%w: image file %T", model.ErrUnsupported, file)
	}
}

// optionFields converts a provider option into form fields. A null value has
// no fields, so it removes a standard member of the same name; arrays repeat
// the member, strings are sent unquoted and other values as JSON text.
func optionFields(member jsonobject.Member) ([]apicall.FormField, error) {
	if bytes.Equal(bytes.TrimSpace(member.Value), []byte("null")) {
		return []apicall.FormField{}, nil
	}

	var items []json.RawMessage
	if err := json.Unmarshal(member.Value, &items); err != nil {
		items = []json.RawMessage{
			member.Value,
		}
	}

	fields := make([]apicall.FormField, 0, len(items))
	for _, item := range items {
		text, err := formText(item)
		if err != nil {
			return nil, fmt.Errorf("%w: provider option %q: %w", model.ErrInvalidArgument, member.Key, err)
		}

		fields = append(fields, apicall.FormField{
			Name:  member.Key,
			Value: text,
		})
	}

	return fields, nil
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
