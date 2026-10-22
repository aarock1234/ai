package model

import (
	"sync"
	"sync/atomic"
)

// singleUseStream guards a stream that can be ranged over once and whose
// resources are released exactly once. Results that own a connection or a
// response body embed it.
type singleUseStream struct {
	release func() error

	started   atomic.Bool
	closeOnce sync.Once
	closeErr  error
}

// claim reports whether this is the first iteration.
func (s *singleUseStream) claim() bool {
	return s.started.CompareAndSwap(false, true)
}

// close calls release once and returns its error on every call.
func (s *singleUseStream) close() error {
	s.closeOnce.Do(func() {
		if s.release != nil {
			s.closeErr = s.release()
		}
	})

	return s.closeErr
}
