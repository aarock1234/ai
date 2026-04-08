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

	result, err := ai.GenerateText(ctx, openai.Chat("gpt-4o"),
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
stream, err := ai.StreamText(ctx, openai.Chat("gpt-4o"),
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

result, err := ai.GenerateObject[Recipe](ctx, openai.Chat("gpt-4o"),
	ai.WithPrompt("A recipe for pasta carbonara"),
)
if err != nil {
	log.Fatal(err)
}

fmt.Println(result.Object.Name)
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

result, err := ai.GenerateText(ctx, openai.Chat("gpt-4o"),
	ai.WithPrompt("What's the weather in Paris?"),
	ai.WithTools(weatherTool),
	ai.WithMaxSteps(3),
)
```

## Embeddings

```go
result, err := ai.Embed(ctx, openai.Embedding("text-embedding-3-small"), "Hello, world!")

// Or batch:
results, err := ai.EmbedMany(ctx, openai.Embedding("text-embedding-3-small"),
	[]string{"first", "second", "third"},
)
```

## Providers

Swap providers without changing your application code.

```go
// OpenAI
model := openai.Chat("gpt-4o")

// OpenRouter (access 200+ models)
model := openrouter.Chat("anthropic/claude-sonnet-4-20250514")

// Any OpenAI-compatible API
model := openai.Chat("my-model", openai.WithBaseURL("https://my-api.example.com/v1"))
```

### Provider Options

```go
// Explicit API key
openai.Chat("gpt-4o", openai.WithAPIKey("sk-..."))

// Custom HTTP client
openai.Chat("gpt-4o", openai.WithHTTPClient(myClient))

// OpenAI organization
openai.Chat("gpt-4o", openai.WithOrganization("org-..."))

// OpenRouter headers
openrouter.Chat("openai/gpt-4o",
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
| `WithMaxSteps(n)` | Max iterations in the tool loop (default: 1) |
| `WithMaxTokens(n)` | Limit response token count |
| `WithTemperature(f)` | Sampling temperature (0.0 -- 2.0) |
| `WithTopP(f)` | Nucleus sampling parameter |
| `WithStopSequences(s...)` | Stop generation at these sequences |
| `WithTimeout(d)` | Overall request timeout |

## Architecture

```
pkg/
├── ai/            Consumer API: GenerateText, StreamText, GenerateObject, Embed
├── model/         Provider contract: Model and EmbeddingModel interfaces
├── provider/
│   ├── openai/    OpenAI provider + shared OpenAI-compatible codec
│   └── openrouter/ OpenRouter provider
├── schema/        JSON Schema generation from Go types
└── apierror/      Shared API error type
```

The SDK is built on three layers:

1. **`ai`** -- The functions you call. Provider-agnostic.
2. **`model`** -- The interfaces providers implement. Four methods for chat, three for embeddings.
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

If your provider uses the OpenAI-compatible API format, import `pkg/provider/openai` and reuse its codec (`EncodeRequest`, `DecodeResponse`, `DoRequest`, `DoStreamRequest`, `DecodeStream`). See `pkg/provider/openrouter/` for a complete example.

## License

MIT
