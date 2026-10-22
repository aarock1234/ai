// Package ai is the application-facing API of the SDK, a Go port of the Vercel
// AI SDK "ai" package. It generates text, structured output, embeddings,
// images, speech, transcripts and rankings with any provider that implements
// the contracts in the model package.
//
// Every top-level function takes a context and one options struct named after
// it, such as Embed and EmbedOptions. Language model prompts use the model
// package's message types directly: model.SystemMessage, model.UserMessage,
// model.AssistantMessage and model.ToolMessage.
//
// Go adaptations of the upstream API:
//
//   - Cancellation, deadlines and the total timeout come from the
//     context.Context passed to each function; there is no abort signal
//     option. Step, chunk and tool timeouts are TimeoutConfiguration
//     durations.
//   - There is no global state: telemetry integrations and the warning
//     logger are per-call options.
//   - Application values that upstream passes to tools as "context" travel
//     as context.Context values.
//   - Tools are passed as an ordered []Tool. NewTool derives the input schema
//     from a Go type and decodes inputs before calling the typed function.
//   - Callbacks are func fields that receive the context and a single event
//     struct.
//   - Errors are typed values that wrap their causes; inspect them with
//     errors.Is and errors.As.
//   - StreamText returns a single-consumer result. Its streams are iterators
//     that run the generation on the consumer's goroutine; Result returns
//     the aggregated outcome once the stream has ended.
//   - Typed outputs are returned as any; OutputAs converts them to their Go
//     type.
//
// Model middleware lives in the middleware subpackage.
package ai
