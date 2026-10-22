package awseventstream

import (
	"encoding/binary"
	"fmt"
)

// headerType is the type tag that precedes a header value.
type headerType byte

// Header value types.
const (
	headerTrue      headerType = 0
	headerFalse     headerType = 1
	headerByte      headerType = 2
	headerShort     headerType = 3
	headerInteger   headerType = 4
	headerLong      headerType = 5
	headerBytes     headerType = 6
	headerString    headerType = 7
	headerTimestamp headerType = 8
	headerUUID      headerType = 9
)

// fixedValueBytes are the value sizes of the fixed-width header types.
var fixedValueBytes = map[headerType]int{
	headerTrue:      0,
	headerFalse:     0,
	headerByte:      1,
	headerShort:     2,
	headerInteger:   4,
	headerLong:      8,
	headerTimestamp: 8,
	headerUUID:      16,
}

// headerCursor reads headers from the header section of a message.
type headerCursor struct {
	data   []byte
	offset int
}

// decodeHeaders returns the string-valued headers of a header section,
// verifying the layout of every header.
func decodeHeaders(data []byte) (map[string]string, error) {
	headers := make(map[string]string)
	cursor := &headerCursor{
		data: data,
	}

	for cursor.offset < len(data) {
		name, err := cursor.name()
		if err != nil {
			return nil, err
		}

		valueType, err := cursor.take(1)
		if err != nil {
			return nil, err
		}

		value, isString, err := cursor.value(headerType(valueType[0]))
		if err != nil {
			return nil, fmt.Errorf("header %q: %w", name, err)
		}

		if isString {
			headers[name] = value
		}
	}

	return headers, nil
}

func (c *headerCursor) name() (string, error) {
	length, err := c.take(1)
	if err != nil {
		return "", err
	}

	name, err := c.take(int(length[0]))
	if err != nil {
		return "", err
	}

	return string(name), nil
}

// value reads one header value. It returns the text of string values and
// reports whether the value is a string.
func (c *headerCursor) value(valueType headerType) (string, bool, error) {
	if size, fixed := fixedValueBytes[valueType]; fixed {
		_, err := c.take(size)

		return "", false, err
	}

	if valueType != headerBytes && valueType != headerString {
		return "", false, fmt.Errorf("%w: unknown header value type %d", ErrInvalidMessage, valueType)
	}

	lengthBytes, err := c.take(2)
	if err != nil {
		return "", false, err
	}

	value, err := c.take(int(binary.BigEndian.Uint16(lengthBytes)))
	if err != nil {
		return "", false, err
	}

	return string(value), valueType == headerString, nil
}

// take returns the next count bytes of the header section.
func (c *headerCursor) take(count int) ([]byte, error) {
	end := c.offset + count
	if end > len(c.data) {
		return nil, fmt.Errorf("%w: truncated headers", ErrInvalidMessage)
	}

	taken := c.data[c.offset:end]
	c.offset = end

	return taken, nil
}
