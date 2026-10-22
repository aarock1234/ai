# AI SDK for Go

[![Go Reference](https://pkg.go.dev/badge/github.com/aarock1234/ai.svg)](https://pkg.go.dev/github.com/aarock1234/ai)

An idiomatic Go port of the [Vercel AI SDK](https://ai-sdk.dev) v7. One API for text generation, streaming, structured output, tools, agents, embeddings, images, video, speech, transcription and reranking across 48 providers, plus the server side of the AI SDK UI protocol, so a Go backend can stream to a `useChat` frontend.

## Install

```sh
go get github.com/aarock1234/ai
```

Requires Go 1.27.3 or later.

## Quick start

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
	provider := openai.New() // reads OPENAI_API_KEY

	result, err := ai.GenerateText(context.Background(), ai.GenerateTextOptions{
		Model: provider.Responses("gpt-5.4"),
		Prompt: ai.Prompt{
			Prompt: "What is the capital of France?",
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(result.Text)
}
```

Every top-level function takes a `context.Context` and one options struct named after it: `GenerateText` and `GenerateTextOptions`, `Embed` and `EmbedOptions`, and so on.

## Packages

| Package | Contents |
| --- | --- |
| `pkg/ai` | Text, streaming, structured output, tools, agents, embeddings, images, speech, transcription, reranking, video, uploads, registry, and the experimental batch, decision, realtime and translation APIs |
| `pkg/ai/middleware` | Language, embedding and image model middleware |
| `pkg/ai/ui` | UI messages and UI message streams for the AI SDK UI (`useChat`) |
| `pkg/model` | The provider specification: model interfaces, messages, content, stream parts and errors |
| `pkg/provider/...` | Provider implementations |

[Additional packages](#additional-packages) cover MCP, coding-agent harnesses, sandboxes, observability and policies.

## Providers

Packages live under `github.com/aarock1234/ai/pkg/provider/`. Create a provider with `New(options...)`, except `openaicompatible.New(name, baseURL)` and `openresponses.New(name, url)`. Credentials are read from the environment when a request is sent; configure providers with options such as `WithAPIKey`, `WithBaseURL`, `WithHTTPClient` and `WithHeaders`. Each provider has typed model constructors, such as `openai.New().Responses("gpt-5.4")` or `anthropic.New().Messages("claude-sonnet-4-5")`, and implements `model.Provider` for lookups by ID.

| Package | Environment | Models |
| --- | --- | --- |
| `alibaba` | `ALIBABA_API_KEY` | Language, embedding, video |
| `amazonbedrock` | `AWS_REGION` with `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_SESSION_TOKEN`, or `AWS_BEARER_TOKEN_BEDROCK` | Language, embedding, image, reranking; `amazonbedrock/anthropic` and `amazonbedrock/mantle` for Claude Messages and Mantle models |
| `anthropic` | `ANTHROPIC_API_KEY` | Language, files, skills, batch, decision |
| `anthropicaws` | `AWS_REGION` with `ANTHROPIC_AWS_API_KEY` or AWS credentials, and `ANTHROPIC_AWS_WORKSPACE_ID` | Language, files, skills |
| `assemblyai` | `ASSEMBLYAI_API_KEY` | Transcription |
| `azure` | `AZURE_API_KEY`, `AZURE_RESOURCE_NAME` | Language, embedding, image, speech, transcription |
| `baseten` | `BASETEN_API_KEY` | Language, embedding |
| `blackforestlabs` | `BFL_API_KEY` | Image, video |
| `bytedance` | `ARK_API_KEY` | Image, video |
| `cartesia` | `CARTESIA_API_KEY` | Speech, transcription, realtime |
| `cerebras` | `CEREBRAS_API_KEY` | Language |
| `cohere` | `COHERE_API_KEY` | Language, embedding, reranking |
| `deepgram` | `DEEPGRAM_API_KEY` | Speech, transcription |
| `deepinfra` | `DEEPINFRA_API_KEY` | Language, embedding, image |
| `deepseek` | `DEEPSEEK_API_KEY` | Language, files |
| `elevenlabs` | `ELEVENLABS_API_KEY` | Speech, transcription |
| `fal` | `FAL_API_KEY` or `FAL_KEY` | Image, speech, transcription, video |
| `fireworks` | `FIREWORKS_API_KEY` | Language, embedding, image |
| `fishaudio` | `FISH_AUDIO_API_KEY` | Speech, transcription |
| `gateway` | `AI_GATEWAY_API_KEY` or `VERCEL_OIDC_TOKEN` | Language, embedding, image, reranking, speech, transcription, video, realtime, decision, batch |
| `gladia` | `GLADIA_API_KEY` | Transcription |
| `gmicloud` | `GMI_CLOUD_APIKEY` | Language |
| `google` | `GOOGLE_GENERATIVE_AI_API_KEY` | Language, embedding, image, speech, transcription, video, files, batch, decision, realtime, speech translation |
| `googlevertex` | `GOOGLE_VERTEX_PROJECT` and `GOOGLE_VERTEX_LOCATION`, with `WithTokenSource` or `GOOGLE_VERTEX_API_KEY` | Language, embedding, image, speech, transcription, video; `googlevertex/anthropic`, `googlevertex/maas` and `googlevertex/xai` for partner models |
| `groq` | `GROQ_API_KEY` | Language, transcription |
| `heygen` | `HEYGEN_API_KEY` | Video |
| `huggingface` | `HUGGINGFACE_API_KEY` | Language |
| `hume` | `HUME_API_KEY` | Speech |
| `klingai` | `KLINGAI_API_KEY`, or `KLINGAI_ACCESS_KEY` and `KLINGAI_SECRET_KEY` | Video |
| `luma` | `LUMA_API_KEY` | Image |
| `minimax` | `MINIMAX_API_KEY` | Language, video |
| `mistral` | `MISTRAL_API_KEY` | Language, embedding, speech, transcription |
| `moonshotai` | `MOONSHOT_API_KEY` | Language |
| `openai` | `OPENAI_API_KEY` | Language, embedding, image, speech, transcription, files, skills, batch, decision, realtime, speech translation |
| `openaicompatible` | none (`WithAPIKey`) | Language, embedding, image |
| `openresponses` | none (`WithAPIKey`) | Language |
| `openrouter` | `OPENROUTER_API_KEY` | Language, embedding |
| `perplexity` | `PERPLEXITY_API_KEY` | Language, embedding, decision |
| `prodia` | `PRODIA_TOKEN` | Language, image, video |
| `quiverai` | `QUIVERAI_API_KEY` | Language, image |
| `replicate` | `REPLICATE_API_TOKEN` | Image, video |
| `revai` | `REVAI_API_KEY` | Transcription |
| `togetherai` | `TOGETHER_API_KEY` | Language, embedding, image, reranking |
| `topaz` | `TOPAZ_API_KEY` | Image, video |
| `typesafeai` | `TYPESAFE_AI_API_KEY` | Decision |
| `voyage` | `VOYAGE_API_KEY` | Embedding, reranking |
| `xai` | `XAI_API_KEY` | Language, image, speech, transcription, video, files, batch, realtime |
| `zai` | `ZAI_API_KEY` | Language |

## Generating text

```go
result, err := ai.GenerateText(ctx, ai.GenerateTextOptions{
	Model: provider.Responses("gpt-5.4"),
	CallSettings: ai.CallSettings{
		MaxOutputTokens: new(500),
	},
	Prompt: ai.Prompt{
		System: "You are a concise assistant.",
		Prompt: "Explain goroutines in one paragraph.",
	},
})
if err != nil {
	return fmt.Errorf("generate text: %w", err)
}

fmt.Println(result.Text, result.FinishReason)
```

Messages are typed per role: `model.SystemMessage`, `model.UserMessage`, `model.AssistantMessage` and `model.ToolMessage`, each accepting only the parts its role allows. Continue a conversation by appending `result.ResponseMessages`:

```go
messages := []model.Message{
	model.UserMessage{
		Content: []model.UserPart{
			model.TextPart{
				Text: "Describe this image.",
			},
			model.FilePart{
				MediaType: "image/png",
				Data: model.FileBytes{
					Data: imageBytes,
				},
			},
		},
	},
}

result, err := ai.GenerateText(ctx, ai.GenerateTextOptions{
	Model: languageModel,
	Prompt: ai.Prompt{
		Messages: messages,
	},
})
if err != nil {
	return fmt.Errorf("describe image: %w", err)
}

messages = append(messages, result.ResponseMessages...)
```

## Streaming

`StreamText` returns a single-consumer stream. Generation runs while you range over one of its iterators and stops when you stop; `Result` returns the aggregated outcome afterwards.

```go
stream, err := ai.StreamText(ctx, ai.StreamTextOptions{
	Model: languageModel,
	Prompt: ai.Prompt{
		Prompt: "Write a haiku about Go.",
	},
})
if err != nil {
	return fmt.Errorf("start stream: %w", err)
}
defer func() { _ = stream.Close() }()

for delta := range stream.TextStream() {
	fmt.Print(delta)
}

result, err := stream.Result()
if err != nil {
	return fmt.Errorf("stream text: %w", err)
}

fmt.Println(result.Usage.TotalTokens != nil)
```

Range over `FullStream` instead to receive every part, including reasoning, tool calls, tool results, step boundaries and errors:

```go
for part := range stream.FullStream() {
	switch part := part.(type) {
	case ai.TextDeltaPart:
		fmt.Print(part.Text)
	case ai.ToolCall:
		fmt.Printf("\ncalling %s\n", part.ToolName)
	case ai.FinishPart:
		fmt.Printf("\nfinished: %s\n", part.FinishReason)
	}
}
```

Set `Transforms: []ai.StreamTransform{ai.SmoothStream(ai.SmoothStreamOptions{})}` to re-chunk text into words.

## Structured output

Set `Output` on `GenerateText` or `StreamText`. The JSON Schema is derived from the Go type, and `OutputAs` returns the validated value.

```go
type Recipe struct {
	Name        string   `json:"name"`
	Ingredients []string `json:"ingredients"`
	Steps       []string `json:"steps" jsonschema:"description=Preparation steps in order"`
}

result, err := ai.GenerateText(ctx, ai.GenerateTextOptions{
	Model: languageModel,
	Prompt: ai.Prompt{
		Prompt: "Give me a recipe for pancakes.",
	},
	Output: ai.ObjectOutput[Recipe](ai.ObjectOutputOptions{
		Name: "recipe",
	}),
})
if err != nil {
	return fmt.Errorf("generate recipe: %w", err)
}

recipe, err := ai.OutputAs[Recipe](result.Output)
if err != nil {
	return fmt.Errorf("read recipe: %w", err)
}
```

`ArrayOutput[T]`, `ChoiceOutput` and `JSONOutput` cover arrays, enumerations and free-form JSON. When streaming, `PartialOutputStream` yields partial values as the JSON arrives and `ElementStream` yields each completed array element. Upstream's deprecated `generateObject` and `streamObject` are not ported; use `Output` instead.

## Tools

`NewTool` derives the input schema from a Go type and decodes the model's input before calling `Execute`. With `StopWhen`, the SDK runs the tool loop: it executes tool calls and sends the results back until the model finishes or a stop condition is met.

```go
type WeatherInput struct {
	City string `json:"city" jsonschema:"description=The city to look up"`
}

weather := ai.NewTool(ai.ToolDefinition[WeatherInput, string]{
	Name:        "get_weather",
	Description: "Get the current weather for a city.",
	Execute: func(ctx context.Context, input WeatherInput, options ai.ToolExecutionOptions) (string, error) {
		return "sunny, 21°C in " + input.City, nil
	},
})

result, err := ai.GenerateText(ctx, ai.GenerateTextOptions{
	Model: languageModel,
	Prompt: ai.Prompt{
		Prompt: "What's the weather in Lisbon?",
	},
	Tools: []ai.Tool{
		weather,
	},
	StopWhen: []ai.StopCondition{
		ai.IsStepCount(5),
	},
})
```

The context passed to `Execute` is cancelled with the generation, and carries any values you attach to it. `ToolExecutionOptions.Preliminary` reports progress while a tool runs. Provider tools are declared with `ai.ProviderExecutedTool`, for example `ai.ProviderExecutedTool(anthropic.WebSearch20250305(anthropic.WebSearchArgs{}))`, and dynamic tools, such as tools discovered from an MCP server, with `ai.DynamicTool`.

### Tool approvals

A tool with `NeedsApproval` stops the loop with an `ai.ToolApprovalRequest` instead of running. Answer it with a `model.ToolApprovalResponsePart` and call the model again: approved calls run, and denied calls are reported to the model.

```go
deleteFile := ai.NewTool(ai.ToolDefinition[DeleteInput, string]{
	Name:        "delete_file",
	Description: "Delete a file.",
	NeedsApproval: func(ctx context.Context, input DeleteInput, options ai.ToolNeedsApprovalOptions) (bool, error) {
		return true, nil
	},
	Execute: deleteFileFromDisk,
})

options := ai.GenerateTextOptions{
	Model: languageModel,
	Prompt: ai.Prompt{
		Messages: messages,
	},
	Tools: []ai.Tool{
		deleteFile,
	},
	StopWhen: []ai.StopCondition{
		ai.IsStepCount(5),
	},
}

result, err := ai.GenerateText(ctx, options)
if err != nil {
	return fmt.Errorf("generate text: %w", err)
}

var responses []model.ToolPart

for _, part := range result.Content {
	request, isRequest := part.(ai.ToolApprovalRequest)
	if !isRequest {
		continue
	}

	responses = append(responses, model.ToolApprovalResponsePart{
		ApprovalID: request.ApprovalID,
		Approved:   askUser(request.ToolCall),
	})
}

options.Messages = append(messages, result.ResponseMessages...)
options.Messages = append(options.Messages, model.ToolMessage{
	Content: responses,
})

result, err = ai.GenerateText(ctx, options)
```

`ToolApproval` replaces the per-tool functions with one policy that can approve, deny or defer to the user. Set `ToolApprovalSecret` to sign approval requests, so that approvals sent back by a client cannot be forged.

## Agents

A `ToolLoopAgent` bundles a model, instructions, tools and stop conditions for reuse. `Generate` and `Stream` accept a prompt or messages per call.

```go
agent := ai.NewToolLoopAgent(ai.ToolLoopAgentSettings{
	ID: "weather",
	GenerateTextOptions: ai.GenerateTextOptions{
		Model: languageModel,
		Prompt: ai.Prompt{
			System: "You answer questions about the weather.",
		},
		Tools: []ai.Tool{
			weather,
		},
	},
})

result, err := agent.Generate(ctx, ai.AgentCallOptions{
	Prompt: "Do I need a jacket in Oslo today?",
})
```

Without `StopWhen`, an agent stops after 20 steps. `ai.Agent` is an interface, so applications can supply their own implementations.

## Embeddings

```go
result, err := ai.EmbedMany(ctx, ai.EmbedManyOptions{
	Model: provider.Embedding("text-embedding-3-small"),
	Values: []string{
		"sunny day at the beach",
		"rainy afternoon in the city",
	},
})
if err != nil {
	return fmt.Errorf("embed values: %w", err)
}

fmt.Println(len(result.Embeddings))
```

`Embed` embeds a single value. `EmbedMany` splits values into batches when the model limits the batch size and runs them concurrently when the model allows it.

## Images, speech, transcription and reranking

```go
images, err := ai.GenerateImage(ctx, ai.GenerateImageOptions{
	Model:  provider.Image("gpt-image-1"),
	Prompt: "A gopher reading a book under a tree.",
	Size:   "1024x1024",
})

speech, err := ai.GenerateSpeech(ctx, ai.GenerateSpeechOptions{
	Model: provider.Speech("gpt-4o-mini-tts"),
	Text:  "Hello from Go!",
	Voice: "alloy",
})

transcript, err := ai.Transcribe(ctx, ai.TranscribeOptions{
	Model: provider.Transcription("gpt-4o-mini-transcribe"),
	Audio: model.FileBytes{
		Data: audio,
	},
})

ranking, err := ai.Rerank(ctx, ai.RerankOptions[string]{
	Model:     cohere.New().Reranking("rerank-v3.5"),
	Query:     "capital of France",
	Documents: documents,
	TopN:      new(3),
})
```

The results expose `images.Images`, `speech.Audio`, `transcript.Text` with timed `Segments`, and `ranking.Ranking` with scores and original indices.

## Middleware

Middleware wraps a model to transform its call options or surround its calls.

```go
wrapped := middleware.WrapLanguageModel(middleware.WrapLanguageModelOptions{
	Model: languageModel,
	Middleware: []middleware.Language{
		middleware.DefaultSettings(model.CallOptions{
			MaxOutputTokens: new(1024),
		}),
		middleware.SimulateStreaming(),
	},
})
```

Built-in middleware includes `DefaultSettings`, `DefaultInstructions`, `ExtractReasoning`, `ExtractJSON`, `SimulateStreaming` and `AddToolInputExamples`. Write your own by setting the `TransformParams`, `WrapGenerate` or `WrapStream` fields of `middleware.Language`.

## Provider registry

A `Registry` resolves combined `provider:model` IDs, so the model can come from configuration.

```go
registry := ai.NewRegistry(ai.RegistryOptions{
	Providers: map[string]model.Provider{
		"anthropic": anthropic.New(),
		"openai":    openai.New(),
	},
})

languageModel, err := registry.LanguageModel("anthropic:claude-sonnet-4-5")
if err != nil {
	return fmt.Errorf("resolve model: %w", err)
}
```

`RegistryOptions.LanguageModelMiddleware` wraps every model the registry returns. `ai.NewCustomProvider` assigns models to aliases, with an optional fallback provider.

## UI message streams

The `ui` package implements the AI SDK UI protocol, so a Go `net/http` handler can serve a `useChat` frontend.

```go
func (h *chatHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	var request struct {
		Messages []ui.Message `json:"messages"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, "invalid request", http.StatusBadRequest)

		return
	}

	messages, err := ui.ValidateMessages(ctx, ui.ValidateMessagesOptions{
		Messages: request.Messages,
	})
	if err != nil {
		http.Error(w, "invalid messages", http.StatusBadRequest)

		return
	}

	prompt, err := ui.ConvertToModelMessages(ctx, messages, ui.ConvertToModelMessagesOptions{})
	if err != nil {
		http.Error(w, "invalid messages", http.StatusBadRequest)

		return
	}

	stream, err := ai.StreamText(ctx, ai.StreamTextOptions{
		Model: h.model,
		Prompt: ai.Prompt{
			Messages: prompt,
		},
	})
	if err != nil {
		http.Error(w, "chat unavailable", http.StatusInternalServerError)

		return
	}
	defer func() { _ = stream.Close() }()

	_ = ui.PipeMessageStreamToResponse(w, ui.MessageStreamResponseOptions{
		Stream: ui.ToMessageStream(ctx, stream.FullStream(), ui.MessageStreamOptions{}),
	})
}
```

`ui.PipeAgentStreamToResponse` does the same for an agent in one call. `ui.CreateMessageStream` composes streams and custom data parts through a `Writer`, and `ui.ReadMessageStream` rebuilds messages from a stream. See [examples/chat-server](examples/chat-server).

## Provider options and metadata

Provider-specific settings are typed structs keyed by provider namespace. Read provider metadata back with `model.ProviderValue`.

```go
result, err := ai.GenerateText(ctx, ai.GenerateTextOptions{
	Model: anthropic.New().Messages("claude-sonnet-4-5"),
	Prompt: ai.Prompt{
		Prompt: "How many prime numbers are there below 100?",
	},
	ProviderOptions: model.ProviderOptions{
		"anthropic": anthropic.MessagesOptions{
			Thinking: &anthropic.ThinkingOptions{
				Type:         anthropic.ThinkingEnabled,
				BudgetTokens: new(4096),
			},
		},
	},
})
if err != nil {
	return fmt.Errorf("generate text: %w", err)
}

metadata, found, err := model.ProviderValue[anthropic.MessagesMetadata](result.ProviderMetadata, "anthropic")
```

For provider-independent reasoning control, set `CallSettings.Reasoning`, for example to `model.ReasoningHigh`.

## Errors

Errors are typed values that wrap their causes. Inspect them with `errors.Is` and `errors.As`:

```go
if apiErr, ok := errors.AsType[*model.APICallError](err); ok {
	fmt.Println(apiErr.StatusCode, apiErr.IsRetryable)
}

if errors.Is(err, model.ErrNoContentGenerated) {
	fmt.Println("the model returned no content")
}
```

Model calls are retried twice by default; set `MaxRetries` to change that.

## Additional packages

| Package | Contents |
| --- | --- |
| `pkg/mcp` | Model Context Protocol client: server tools as `[]ai.Tool`, resources, prompts, elicitation, Streamable HTTP, SSE and stdio transports, OAuth |
| `pkg/harness` | Drives coding-agent runtimes inside a sandbox. Adapters: `harness/acp`, `claudecode`, `codex`, `cursor`, `deepagents`, `fx`, `githubcopilot`, `grokbuild` and `opencode` |
| `pkg/sandbox` | Sandbox session contract, with `sandbox/vercel` (Vercel Sandbox microVMs) and `sandbox/local` (a local shell that is **not isolated**); the `sandboxshell` module adds an isolated in-memory bash sandbox |
| `pkg/tools/bash` | A bash tool that runs commands in any sandbox session |
| `pkg/devtools` | Records calls for local debugging; `go run github.com/aarock1234/ai/pkg/devtools/cmd/devtools` serves the viewer |
| `pkg/policyopa` | Open Policy Agent tool approval policies and capability middleware |

OpenTelemetry, code mode, the terminal UI, the OPA WebAssembly client and the in-memory shell sandbox are nested modules, so the core module does not depend on their libraries. Install them separately:

```sh
go get github.com/aarock1234/ai/otel           # OpenTelemetry spans: otel.New, otel.NewLegacy
go get github.com/aarock1234/ai/codemode       # TypeScript code mode over your tools: codemode.NewTool
go get github.com/aarock1234/ai/tui            # Terminal UI for an ai.Agent: tui.RunAgent
go get github.com/aarock1234/ai/policyopawasm  # In-process OPA WebAssembly policies: policyopawasm.New
go get github.com/aarock1234/ai/sandboxshell   # Isolated in-memory bash sandbox: sandboxshell.New
```

Combine MCP tools, a sandboxed bash tool and OpenTelemetry in one call:

```go
transport, err := mcp.NewHTTPTransport(mcp.HTTPTransportOptions{
	URL: "https://mcp.example.com/mcp",
})
if err != nil {
	return fmt.Errorf("create transport: %w", err)
}

client, err := mcp.NewClient(ctx, mcp.ClientOptions{
	Transport: transport,
})
if err != nil {
	return fmt.Errorf("connect to mcp server: %w", err)
}
defer func() {
	err = errors.Join(err, client.Close(context.WithoutCancel(ctx)))
}()

tools, err := client.Tools(ctx, mcp.ToolsOptions{})
if err != nil {
	return fmt.Errorf("list mcp tools: %w", err)
}

session, err := local.New(ctx, local.Options{})
if err != nil {
	return fmt.Errorf("create sandbox: %w", err)
}
defer func() {
	err = errors.Join(err, session.Destroy(context.WithoutCancel(ctx)))
}()

bashTool, err := bash.New(ctx, bash.Options{
	Sandbox: session,
})
if err != nil {
	return fmt.Errorf("create bash tool: %w", err)
}

result, err := ai.GenerateText(ctx, ai.GenerateTextOptions{
	Model: languageModel,
	Prompt: ai.Prompt{
		Prompt: "Summarize the files in the working directory.",
	},
	Tools: append(tools, bashTool),
	StopWhen: []ai.StopCondition{
		ai.IsStepCount(10),
	},
	Telemetry: ai.TelemetryOptions{
		Integrations: []ai.Telemetry{
			otel.New(otel.Options{}),
		},
	},
})
```

## Go adaptations

The API follows the TypeScript SDK closely, with these deliberate differences:

- **`context.Context` instead of abort signals.** Cancellation and deadlines come from the context passed to each function. Application values that upstream passes to tools as `context` travel as context values.
- **Single-consumer, pull-based streams.** `StreamText` streams are Go iterators that run the generation on the consumer's goroutine. Range over exactly one stream, once; breaking out of the loop stops the generation and releases the provider connection. There is no tee'd stream fan-out.
- **Typed provider options.** `model.ProviderOptions` values are each provider's options struct, and `model.ProviderValue` decodes options and metadata into a typed struct.
- **Typed per-role messages.** Each message role is its own type and accepts only the parts its role allows, so many invalid prompts fail to compile.
- **Options structs and callbacks.** Each function takes one options struct; callbacks are func fields that receive the context and a single event struct.
- **Typed outputs.** Parsed outputs are returned as `any`; `ai.OutputAs[T]` converts them to their Go type.

## Examples

| Example                                             | Shows                                           |
| --------------------------------------------------- | ----------------------------------------------- |
| [generate-text](examples/generate-text)             | Generating text with usage                      |
| [stream-text](examples/stream-text)                 | Streaming text deltas                           |
| [conversation](examples/conversation)               | Multi-turn conversation history                 |
| [generate-object](examples/generate-object)         | Structured output with `ObjectOutput`           |
| [stream-object](examples/stream-object)             | Streaming partial structured output             |
| [tool-calling](examples/tool-calling)               | Typed tools and the tool loop                   |
| [agent](examples/agent)                             | A reusable `ToolLoopAgent`                      |
| [multi-provider](examples/multi-provider)           | Selecting models with a registry                |
| [embeddings](examples/embeddings)                   | Embeddings and cosine similarity                |
| [rerank](examples/rerank)                           | Reranking documents                             |
| [generate-image](examples/generate-image)           | Image generation with provider options          |
| [generate-speech](examples/generate-speech)         | Text to speech                                  |
| [transcribe](examples/transcribe)                   | Speech to text                                  |
| [chat-server](examples/chat-server)                 | Streaming to `useChat` from a `net/http` server |
| [mcp](examples/mcp)                                 | Listing an MCP server's tools over stdio        |

Run an example with `go run ./examples/<name>` after setting the provider's API key.

## Migrating

See [docs/migration.md](docs/migration.md) for changes from earlier versions of this SDK.

## License

This SDK is licensed under the [Apache License 2.0](LICENSE), the same license as the [Vercel AI SDK](https://github.com/vercel/ai) it ports. See [NOTICE](NOTICE) for attribution and the embedded upstream files.
