// Package ui is the server side of the AI SDK UI protocols, a Go port of the
// Vercel AI SDK "ui" and "ui-message-stream" modules. Its JSON matches the
// upstream wire format, so Go servers interoperate with the TypeScript
// useChat frontend.
//
// Messages and their parts are Message and the Part types. A response
// streams Chunk values, encoded as Server-Sent Events by
// PipeMessageStreamToResponse. ToMessageStream converts a StreamText stream
// into chunks, CreateMessageStream composes streams with a Writer, and
// ReadMessageStream rebuilds messages from chunks. ConvertToModelMessages
// turns validated UI messages into a model prompt.
//
// Go adaptations of the upstream API:
//
//   - Streams are iterators: iter.Seq2[Chunk, error], where an error ends
//     the stream. Breaking out of the range releases the source stream.
//   - Dynamic JSON values, such as message metadata, data part payloads and
//     tool inputs and outputs, are json.RawMessage.
//   - Static and dynamic tool parts share the ToolPart type; Dynamic selects
//     the "dynamic-tool" encoding.
//   - HTTP responses are written to an http.ResponseWriter or returned as an
//     *http.Response. Status text cannot be customized, and the copy of the
//     SSE stream that upstream passes to consumeSseStream is written to an
//     io.Writer.
//   - Optional booleans that upstream defaults to true are *bool.
package ui
