package mcp

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
)

// ContentType is the type of a content block in tool results and prompt
// messages.
type ContentType string

// Content types the client models. Blocks of other types, such as audio,
// are kept with their raw encoding.
const (
	ContentTypeText         ContentType = "text"
	ContentTypeImage        ContentType = "image"
	ContentTypeResource     ContentType = "resource"
	ContentTypeResourceLink ContentType = "resource_link"
)

// Content is a content block. The members used depend on Type: Text for
// text; Data and MIMEType for image; Resource for an embedded resource; URI,
// Name, Description and MIMEType for a resource link.
type Content struct {
	Type ContentType

	Text string

	// Data is the decoded content of an image; it is base64 on the wire.
	Data     []byte
	MIMEType string

	Resource *ResourceContents

	URI         string
	Name        string
	Description string

	// Raw is the block as received, including members not modeled here. It
	// is encoded unchanged when set.
	Raw json.RawMessage
}

// contentWire is the encoding of a content block. Pointers distinguish
// absent members from empty ones.
type contentWire struct {
	Type        ContentType       `json:"type"`
	Text        *string           `json:"text,omitempty"`
	Data        *[]byte           `json:"data,omitempty"`
	MIMEType    *string           `json:"mimeType,omitempty"`
	Resource    *ResourceContents `json:"resource,omitempty"`
	URI         *string           `json:"uri,omitempty"`
	Name        *string           `json:"name,omitempty"`
	Description *string           `json:"description,omitempty"`
}

// UnmarshalJSON decodes and validates a content block. Known types must have
// their required members.
func (c *Content) UnmarshalJSON(data []byte) error {
	var wire contentWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}

	if err := validateContent(wire); err != nil {
		return err
	}

	*c = Content{
		Type:        wire.Type,
		Text:        valueOf(wire.Text),
		Data:        bytesOf(wire.Data),
		MIMEType:    valueOf(wire.MIMEType),
		Resource:    wire.Resource,
		URI:         valueOf(wire.URI),
		Name:        valueOf(wire.Name),
		Description: valueOf(wire.Description),
		Raw:         bytes.Clone(data),
	}

	return nil
}

// MarshalJSON encodes Raw when set and the modeled members otherwise.
func (c Content) MarshalJSON() ([]byte, error) {
	if len(c.Raw) > 0 {
		return c.Raw, nil
	}

	wire := contentWire{
		Type:     c.Type,
		Resource: c.Resource,
	}

	switch c.Type {
	case ContentTypeText:
		wire.Text = &c.Text
	case ContentTypeImage:
		wire.Data = &c.Data
		wire.MIMEType = &c.MIMEType
	case ContentTypeResourceLink:
		wire.URI = &c.URI
		wire.Name = &c.Name
		wire.Description = optionalString(c.Description)
		wire.MIMEType = optionalString(c.MIMEType)
	}

	return json.Marshal(wire)
}

func validateContent(wire contentWire) error {
	switch wire.Type {
	case "":
		return errors.New("content block must have a type")
	case ContentTypeText:
		if wire.Text == nil {
			return errors.New("text content must have text")
		}
	case ContentTypeImage:
		if wire.Data == nil || wire.MIMEType == nil {
			return errors.New("image content must have data and mimeType")
		}
	case ContentTypeResource:
		if wire.Resource == nil {
			return errors.New("resource content must have a resource")
		}
	case ContentTypeResourceLink:
		if wire.URI == nil || wire.Name == nil {
			return errors.New("resource_link content must have a uri and a name")
		}
	}

	return nil
}

// ResourceContents is the content of a resource: text, or a binary blob
// when Blob is non-nil.
type ResourceContents struct {
	URI      string
	Name     string
	Title    string
	MIMEType string

	Text string

	// Blob is the decoded binary content; it is base64 on the wire.
	Blob []byte

	Meta map[string]json.RawMessage
}

// IsBlob reports whether the contents are binary.
func (r ResourceContents) IsBlob() bool {
	return r.Blob != nil
}

type resourceContentsWire struct {
	URI      string                     `json:"uri"`
	Name     string                     `json:"name,omitempty"`
	Title    string                     `json:"title,omitempty"`
	MIMEType string                     `json:"mimeType,omitempty"`
	Text     *string                    `json:"text,omitempty"`
	Blob     *[]byte                    `json:"blob,omitempty"`
	Meta     map[string]json.RawMessage `json:"_meta,omitempty"`
}

// UnmarshalJSON decodes text or blob resource contents.
func (r *ResourceContents) UnmarshalJSON(data []byte) error {
	var wire resourceContentsWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}

	if err := requireFields(data, "uri"); err != nil {
		return fmt.Errorf("resource contents: %w", err)
	}

	if wire.Text == nil && wire.Blob == nil {
		return errors.New("resource contents must have text or a blob")
	}

	*r = ResourceContents{
		URI:      wire.URI,
		Name:     wire.Name,
		Title:    wire.Title,
		MIMEType: wire.MIMEType,
		Text:     valueOf(wire.Text),
		Blob:     bytesOf(wire.Blob),
		Meta:     wire.Meta,
	}

	return nil
}

// MarshalJSON encodes the blob when present and the text otherwise.
func (r ResourceContents) MarshalJSON() ([]byte, error) {
	wire := resourceContentsWire{
		URI:      r.URI,
		Name:     r.Name,
		Title:    r.Title,
		MIMEType: r.MIMEType,
		Meta:     r.Meta,
	}

	if r.IsBlob() {
		wire.Blob = &r.Blob
	} else {
		wire.Text = &r.Text
	}

	return json.Marshal(wire)
}

func valueOf(value *string) string {
	if value == nil {
		return ""
	}

	return *value
}

// bytesOf returns decoded base64 bytes, keeping a present but empty value
// non-nil.
func bytesOf(value *[]byte) []byte {
	if value == nil {
		return nil
	}

	if *value == nil {
		return []byte{}
	}

	return *value
}

func optionalString(value string) *string {
	if value == "" {
		return nil
	}

	return &value
}
