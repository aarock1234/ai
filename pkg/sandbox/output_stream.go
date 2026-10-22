package sandbox

import (
	"bytes"
	"io"
	"sync"
)

// OutputStream is an in-memory pipe without a size limit, for the output
// streams of a Process. Writes never block, so a producer that feeds stdout
// and stderr from one source cannot stall on a stream nobody reads. Reads
// block until data arrives or the stream is closed. The zero value is ready
// to use; it must not be copied after first use.
type OutputStream struct {
	mu     sync.Mutex
	cond   *sync.Cond
	buffer bytes.Buffer
	closed bool
	err    error
}

// Write appends data to the stream. It fails with io.ErrClosedPipe after the
// stream was closed.
func (s *OutputStream) Write(data []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return 0, io.ErrClosedPipe
	}

	s.buffer.Write(data)
	s.signal()

	return len(data), nil
}

// Read reads buffered data, blocking until data is available. After the
// stream is closed and drained it returns io.EOF, or the error passed to
// CloseWithError.
func (s *OutputStream) Read(data []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for s.buffer.Len() == 0 && !s.closed {
		s.wait()
	}

	if s.buffer.Len() > 0 {
		return s.buffer.Read(data)
	}

	if s.err != nil {
		return 0, s.err
	}

	return 0, io.EOF
}

// Close ends the stream. Readers receive the buffered data and then io.EOF.
func (s *OutputStream) Close() error {
	return s.CloseWithError(nil)
}

// CloseWithError ends the stream. Readers receive the buffered data and then
// err, or io.EOF when err is nil. Only the first close takes effect.
func (s *OutputStream) CloseWithError(err error) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil
	}

	s.closed = true
	s.err = err
	s.signal()

	return nil
}

// signal wakes blocked readers. The caller holds mu.
func (s *OutputStream) signal() {
	if s.cond != nil {
		s.cond.Broadcast()
	}
}

// wait blocks until the stream changes. The caller holds mu.
func (s *OutputStream) wait() {
	if s.cond == nil {
		s.cond = sync.NewCond(&s.mu)
	}

	s.cond.Wait()
}
