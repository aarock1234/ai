# AI SDK for Go

[![Go Reference](https://pkg.go.dev/badge/github.com/aarock1234/ai.svg)](https://pkg.go.dev/github.com/aarock1234/ai)
[![Go Report Card](https://goreportcard.com/badge/github.com/aarock1234/ai)](https://goreportcard.com/report/github.com/aarock1234/ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A unified Go SDK for working with large language models. Write provider-agnostic code once, swap models with a single line.

## Install

```sh
go get github.com/aarock1234/ai
```

Requires Go 1.23+.

## Quick Start

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

func main() {
	ctx := context.Background()

	result, err := ai.GenerateText(ctx, openai.Chat("gpt-5.4"),
		ai.WithPrompt("What is the capital of France?"),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(result.Text)
}
```

## Streaming

```go
stream, err := ai.StreamText(ctx, openai.Chat("gpt-5.4"),
	ai.WithPrompt("Tell me a story"),
)
if err != nil {
	log.Fatal(err)
}

for delta := range stream.Text() {
	fmt.Print(delta)
}
```

## Structured Output

Generate type-safe structured data. The SDK derives the JSON Schema from your Go type automatically.

```go
type Recipe struct {
	Name        string   `json:"name"`
	Ingredients []string `json:"ingredients"`
	Steps       []string `json:"steps"`
}

result, err := ai.GenerateObject[Recipe](ctx, openai.Chat("gpt-5.4"),
	ai.WithPrompt("A recipe for pasta carbonara"),
)
if err != nil {
	log.Fatal(err)
}

fmt.Println(result.Object.Name)
```

For streaming structured output:

```go
stream, err := ai.StreamObject[Recipe](ctx, openai.Chat("gpt-5.4"),
	ai.WithPrompt("A recipe for pasta carbonara"),
)
if err != nil {
	log.Fatal(err)
}

for delta := range stream.JSON() {
	fmt.Print(delta)
}

result, err := stream.Result()
```

## Tool Calling

Define tools and let the SDK run the automatic tool loop.

```go
weatherTool := ai.Tool{
	Name:        "get_weather",
	Description: "Get the current weather for a city",
	InputSchema: ai.SchemaFor[WeatherInput](),
	Execute: func(ctx context.Context, input json.RawMessage) (string, error) {
		var params WeatherInput
		json.Unmarshal(input, &params)
		return fmt.Sprintf("72°F and sunny in %s", params.City), nil
	},
}

result, err := ai.GenerateText(ctx, openai.Chat("gpt-5.4"),
	ai.WithPrompt("What's the weather in Paris?"),
	ai.WithTools(weatherTool),
	ai.WithMaxSteps(3),
)
```

## Multimodal Output

Results carry every content segment the model produced, not just text. Generated files (images, audio), source citations, and reasoning blocks all flow through the normalized `GenerateResult` alongside `Text`.

### Image Output

OpenRouter returns image-generation models' output as base64 data URLs on `message.images[]`. The SDK decodes those into `File` entries:

```go
result, err := ai.GenerateText(ctx, openrouter.Chat("google/gemini-2.5-flash-image-preview"),
	ai.WithPrompt("A watercolor lighthouse at sunrise"),
	ai.WithModalities(model.ModalityText, model.ModalityImage),
)
if err != nil {
	log.Fatal(err)
}

for _, file := range result.Files {
	// file.Data is already base64-decoded bytes.
	os.WriteFile("out.png", file.Data, 0o644)
	fmt.Println("saved", file.MIMEType)
}
```

For streaming, iterate files as they arrive:

```go
stream, _ := ai.StreamText(ctx, model,
	ai.WithPrompt(prompt),
	ai.WithModalities(model.ModalityText, model.ModalityImage),
)
for file := range stream.Files() {
	save(file)
}
```

### Audio Output

OpenAI `gpt-audio` and friends emit audio via `message.audio.data`. Request it with `WithAudio`:

```go
result, err := ai.GenerateText(ctx, openai.Chat("gpt-audio"),
	ai.WithPrompt("Say hello in a warm voice."),
	ai.WithModalities(model.ModalityText, model.ModalityAudio),
	ai.WithAudio("alloy", "wav"),
)
// result.Files[0] holds the audio bytes, result.Text holds the transcript.
```

### Reasoning Output

Anthropic extended thinking, OpenAI o-series, and OpenRouter reasoning models populate `Reasoning` blocks on the result. Signatures and formats are preserved so blocks round-trip back to the provider on follow-up turns:

```go
result, err := ai.GenerateText(ctx, anthropic.Chat("claude-opus-4-5"),
	ai.WithPrompt("Solve this step-by-step..."),
	ai.WithReasoning(model.ReasoningOptions{Effort: "high"}),
)

fmt.Println("reasoning:", result.ReasoningText())
fmt.Println("answer:", result.Text)

for _, block := range result.Reasoning {
	if block.Redacted {
		continue // encrypted/summary block; keep for round-tripping
	}
	fmt.Println(block.Format, block.Signature, block.Text)
}
```

When running a tool loop, the SDK automatically re-sends reasoning blocks back to the provider on subsequent turns so thinking state carries forward.

### Source Citations

Web search tools (OpenAI `gpt-5-search`, OpenRouter models with web grounding) emit `url_citation` annotations that surface as `Source` entries:

```go
for _, src := range result.Sources {
	fmt.Printf("%s: %s [%d-%d]\n", src.Title, src.URL, src.StartIndex, src.EndIndex)
}
```

### File Inputs

Send PDFs, audio, or arbitrary files as user input via `FilePart`:

```go
pdfBytes, _ := os.ReadFile("doc.pdf")

result, err := ai.GenerateText(ctx, anthropic.Chat("claude-opus-4-5"),
	ai.WithMessages(ai.Message{
		Role: ai.RoleUser,
		Parts: []ai.Part{
			ai.TextPart{Text: "Summarize this document."},
			ai.FilePart{Data: pdfBytes, MIMEType: "application/pdf"},
		},
	}),
)
```

Image inputs use the existing `ai.ImagePart` for clarity; other MIME types use `FilePart`.

## Embeddings

```go
result, err := ai.Embed(ctx, openai.Embedding("text-embedding-3-small"), "Hello, world!")

// Or batch:
results, err := ai.EmbedMany(ctx, openai.Embedding("text-embedding-3-small"),
	[]string{"first", "second", "third"},
)
```

## Images (dedicated endpoint)

For providers with a separate image-generation endpoint (e.g. OpenAI `/images`):

```go
result, err := ai.GenerateImage(ctx, openai.Image("gpt-image-1"), model.ImageRequest{
	Prompt:       "A watercolor lighthouse at sunrise",
	Size:         "1024x1024",
	OutputFormat: "png",
})
```

For chat-based image generation (OpenRouter, Gemini), use `ai.GenerateText` with `WithModalities` — see [Image Output](#image-output) above.

## Audio

```go
speech, err := ai.GenerateSpeech(ctx, openai.Speech("gpt-4o-mini-tts"), model.SpeechRequest{
	Input:          "Hello from the AI SDK for Go.",
	Voice:          "alloy",
	ResponseFormat: "mp3",
})

transcript, err := ai.Transcribe(ctx, openai.Transcription("gpt-4o-mini-transcribe"), model.TranscriptionRequest{
	Audio: model.AudioInput{
		Data:     audioBytes,
		Filename: "sample.mp3",
	},
})
```

For chat-based audio generation (OpenAI `gpt-audio`), use `ai.GenerateText` with `WithAudio` — see [Audio Output](#audio-output) above.

## Reranking

```go
result, err := ai.Rerank(ctx, cohere.Rerank("rerank-v3.5"), model.RerankRequest{
	Query:     "golang concurrency",
	Documents: docs,
})
```

## Providers

Swap providers without changing your application code.

```go
// OpenAI
model := openai.Chat("gpt-5.4")

// OpenRouter (access 200+ models)
model := openrouter.Chat("anthropic/claude-sonnet-4-20250514")

// Anthropic
model := anthropic.Chat("claude-sonnet-4-5")

// Cohere reranking
ranker := cohere.Rerank("rerank-v3.5")

// Any OpenAI-compatible API
model := openai.Chat("my-model", openai.WithBaseURL("https://my-api.example.com/v1"))
```

### Provider Options

```go
// Explicit API key
openai.Chat("gpt-5.4", openai.WithAPIKey("sk-..."))

// Custom HTTP client
openai.Chat("gpt-5.4", openai.WithHTTPClient(myClient))

// OpenAI organization
openai.Chat("gpt-5.4", openai.WithOrganization("org-..."))

// OpenRouter headers
openrouter.Chat("openai/gpt-5.4",
	openrouter.WithHTTPReferer("https://myapp.com"),
	openrouter.WithSiteURL("My App"),
)
```

By default, providers read API keys from the environment (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`).

## Provider Registry

For dynamic, config-driven provider selection:

```go
registry := provider.NewRegistry()
registry.Register("openai", openai.New())
registry.Register("openrouter", openrouter.New())

p, ok := registry.Get(os.Getenv("AI_PROVIDER"))
if !ok {
	log.Fatal("unknown provider")
}

result, err := ai.GenerateText(ctx, p.Chat(os.Getenv("AI_MODEL")),
	ai.WithPrompt("Hello"),
)
```

## Generation Options

| Option | Description |
|---|---|
| `WithPrompt(s)` | Set the user prompt |
| `WithSystem(s)` | Set the system message |
| `WithMessages(m...)` | Provide full conversation history |
| `WithTools(t...)` | Register tools for the model |
| `WithActiveTools(names...)` | Limit the active tool subset |
| `WithToolChoice(choice)` | Control tool selection policy |
| `WithMaxSteps(n)` | Max iterations in the tool loop (default: 1) |
| `WithMaxRetries(n)` | Retry retryable provider failures |
| `WithMaxTokens(n)` | Limit response token count |
| `WithTemperature(f)` | Sampling temperature (0.0 -- 2.0) |
| `WithTopP(f)` | Nucleus sampling parameter |
| `WithTopK(n)` | Top-k sampling parameter |
| `WithPresencePenalty(f)` | Penalize repeated topics |
| `WithFrequencyPenalty(f)` | Penalize repeated tokens |
| `WithSeed(n)` | Request deterministic sampling when supported |
| `WithStopSequences(s...)` | Stop generation at these sequences |
| `WithHeaders(h)` | Attach per-request HTTP headers |
| `WithTimeout(d)` | Overall request timeout |
| `WithModalities(m...)` | Request `text`, `image`, and/or `audio` output |
| `WithAudio(voice, format)` | Configure audio output (voice, encoding) |
| `WithReasoning(opts)` | Configure reasoning budget for reasoning models |
| `WithReasoningEffort(level)` | Shortcut for `"low" / "medium" / "high"` reasoning |
| `WithPrediction(content)` | Attach predicted output for faster generation |
| `WithProviderOption(p, k, v)` | Provider-specific request option passthrough |

## Result Shape

`GenerateResult` and `Step` expose every content segment the provider returned:

| Field | Type | Populated when |
|---|---|---|
| `Text` | `string` | Always; concatenated plain-text output |
| `Reasoning` | `[]Reasoning` | Model emitted thinking / reasoning_details blocks |
| `Files` | `[]File` | Model emitted images or audio |
| `Sources` | `[]Source` | Model cited web pages or documents |
| `ToolCalls` | `[]ToolCall` | Model requested tool invocations |
| `ToolResults` | `[]ToolResult` | Tool loop executed tools |
| `Usage` | `Usage` | Always; input / output / reasoning / audio / image breakdown |
| `Warnings` | `[]string` | Provider surfaced non-fatal issues (refusals, unsupported settings) |
| `Metadata` | `ProviderMetadata` | Always; request id, model id, provider-specific fields |
| `Steps` | `[]Step` | `WithMaxSteps > 1` tool loop ran |

`result.ReasoningText()` concatenates non-redacted reasoning for quick display. `Reasoning[i].Signature` and `Signature.Format` round-trip back to providers that verify prior thinking on follow-up turns — the SDK does this automatically in the tool loop.

## Architecture

```
pkg/
├── ai/            Consumer API: text, object, image, audio, rerank, middleware
├── model/         Provider contracts and normalized request/response types
└── provider/
    ├── anthropic/ Anthropic Messages API provider
    ├── cohere/    Cohere reranking provider
    ├── openai/    OpenAI provider + shared OpenAI-compatible codec
    └── openrouter/ OpenRouter provider

internal/
├── apierror/      Shared API error type
├── provideropts/  Provider-specific request option passthrough
├── schema/        JSON Schema generation from Go types
└── sse/           Server-sent events parser
```

The SDK is built on three layers:

1. **`ai`** -- The functions you call. Provider-agnostic.
2. **`model`** -- The interfaces providers implement plus normalized request/response/content types.
3. **`provider/*`** -- Concrete implementations. OpenRouter reuses the OpenAI codec since they share the same wire format.

## Adding a Provider

Implement `model.Model`:

```go
type Model interface {
	Provider() string
	ModelID() string
	DoGenerate(ctx context.Context, req *Request) (*Response, error)
	DoStream(ctx context.Context, req *Request) (*StreamResponse, error)
}
```

Populate `Response.Content` with any of `TextContent`, `ReasoningContent`, `FileContent`, `SourceContent`, `ToolCallContent`, or `ToolResultContent` depending on what the provider emits. Corresponding stream chunks (`TextDelta`, `ReasoningDelta`, `FileChunk`, `SourceChunk`, `ToolCallDelta`, `FinishChunk`, `UsageChunk`, `ErrorChunk`) flow through `Model.DoStream`.

If your provider uses the OpenAI-compatible API format, import `pkg/provider/openai` and reuse its codec (`EncodeRequest`, `DecodeResponse`, `DoRequest`, `DoStreamRequest`, `DecodeStream`). See `pkg/provider/openrouter/` for a complete example.

## License

MIT
