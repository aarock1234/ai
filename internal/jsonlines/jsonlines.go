// Package jsonlines reads the JSON Lines result files of provider batches.
// A Reader owns one response body at a time and guards it against a Close
// that races with reading, so that no body outlives the result stream.
package jsonlines

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"iter"
	"sync"

	"github.com/aarock1234/ai/pkg/model"
)

// DefaultMaxLineBytes bounds one line when the caller configures no limit.
const DefaultMaxLineBytes = 64 << 20

// initialBufferBytes is the initial capacity of the line buffer.
const initialBufferBytes = 64 << 10

// ErrClosed reports a body handed to a reader that was already closed.
var ErrClosed = errors.New("json lines reader closed")

// Reader yields the nonblank lines of the bodies it is given, one body at a
// time. Its methods are safe for concurrent use with Close.
type Reader struct {
	maxLineBytes int

	mu     sync.Mutex
	body   io.ReadCloser
	closed bool
}

// NewReader returns a reader that rejects lines longer than maxLineBytes.
// A limit that is not positive selects DefaultMaxLineBytes.
func NewReader(maxLineBytes int) *Reader {
	if maxLineBytes <= 0 {
		maxLineBytes = DefaultMaxLineBytes
	}

	return &Reader{
		maxLineBytes: maxLineBytes,
	}
}

// Open hands body to the reader, which closes it after Lines or Close. A
// previous body is closed first. When the reader is already closed, Open
// closes body and returns ErrClosed.
func (r *Reader) Open(body io.ReadCloser) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.closed {
		_ = body.Close()

		return ErrClosed
	}

	if r.body != nil {
		_ = r.body.Close()
	}

	r.body = body

	return nil
}

// Lines yields every nonblank line of the open body with surrounding white
// space trimmed, then closes the body. A read failure ends the sequence with
// an error; a line over the limit wraps model.ErrInvalidResponse. The
// yielded bytes are valid only until the next iteration.
func (r *Reader) Lines() iter.Seq2[[]byte, error] {
	return func(yield func([]byte, error) bool) {
		body := r.current()
		if body == nil {
			return
		}

		defer r.release()

		scanner := bufio.NewScanner(body)
		scanner.Buffer(make([]byte, 0, initialBufferBytes), r.maxLineBytes)

		for scanner.Scan() {
			line := bytes.TrimSpace(scanner.Bytes())
			if len(line) == 0 {
				continue
			}

			if !yield(line, nil) {
				return
			}
		}

		if err := scanner.Err(); err != nil {
			yield(nil, r.scanError(err))
		}
	}
}

// Close closes the open body and makes later calls to Open fail.
func (r *Reader) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.closed = true

	return r.closeBody()
}

func (r *Reader) current() io.ReadCloser {
	r.mu.Lock()
	defer r.mu.Unlock()

	return r.body
}

func (r *Reader) release() {
	r.mu.Lock()
	defer r.mu.Unlock()

	_ = r.closeBody()
}

// closeBody closes the open body. The caller holds mu.
func (r *Reader) closeBody() error {
	if r.body == nil {
		return nil
	}

	err := r.body.Close()
	r.body = nil

	return err
}

func (r *Reader) scanError(err error) error {
	if errors.Is(err, bufio.ErrTooLong) {
		return fmt.Errorf(
			"%w: JSON Lines response exceeded maximum line size of %d bytes",
			model.ErrInvalidResponse,
			r.maxLineBytes,
		)
	}

	return fmt.Errorf("read json lines: %w", err)
}
