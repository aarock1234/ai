package middleware

import "github.com/aarock1234/ai/pkg/model"

// partEmitter forwards transformed stream parts to a consumer and remembers
// when the consumer stops iterating.
type partEmitter struct {
	yield   func(model.StreamPart) bool
	stopped bool
}

func (e *partEmitter) emit(part model.StreamPart) {
	if e.stopped {
		return
	}

	if !e.yield(part) {
		e.stopped = true
	}
}

// partTransformer rewrites a stream one part at a time. It may emit any
// number of parts for each input part.
type partTransformer interface {
	transform(part model.StreamPart, out *partEmitter)
}

// transformStream returns a stream that passes every part of source through
// transformer. Closing the returned stream, or ending its iteration early,
// releases source.
func transformStream(source *model.StreamResult, transformer partTransformer) *model.StreamResult {
	parts := func(yield func(model.StreamPart) bool) {
		out := &partEmitter{
			yield: yield,
		}

		for part := range source.Parts() {
			transformer.transform(part, out)

			if out.stopped {
				return
			}
		}
	}

	transformed := model.NewStreamResult(parts, source.Close)
	transformed.Request = source.Request
	transformed.Response = source.Response

	return transformed
}
