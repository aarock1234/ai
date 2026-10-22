# Migrating from v0.3.0

The SDK was rewritten as a port of `ai@7.0.146`. Almost every public API changed. This guide maps the v0.3.0-era API (the last tagged release) to the new one. Every snippet below compiles against its version.

If you built against an untagged snapshot of the rewrite, also read [Changes since the first rewrite snapshots](#changes-since-the-first-rewrite-snapshots). [New packages](#new-packages) lists the APIs that have no v0.3.0 equivalent.

## Summary

| Area | v0.3.0 | Now |
| --- | --- | --- |
| Model constructors | `openai.Chat(id, opts...)` package functions | `openai.New(opts...).Chat(id)` provider methods |
| Language model interface | `model.Model` with `DoGenerate` and `DoStream` | `model.LanguageModel` with `Generate`, `Stream` and `SupportedURLs` |
| Request type | `*model.Request` | `model.CallOptions` |
| Call options | Functional options (`ai.WithPrompt`) | One options struct per function (`ai.GenerateTextOptions`) |
| Messages | `ai.Message{Role, Content, Parts}` | `model.SystemMessage`, `UserMessage`, `AssistantMessage`, `ToolMessage` |
| Images in prompts | `ai.ImagePart` | `model.FilePart` with an `image/*` media type |
| Tools | `Execute(ctx, json.RawMessage) (string, error)` | `ai.NewTool[Input, Output]`, or `Execute(ctx, input, options) (any, error)` |
| Structured output | `ai.GenerateObject[T]`, `ai.StreamObject[T]` | `GenerateText` or `StreamText` with `Output: ai.ObjectOutput[T](...)` |
| Provider options | `ai.WithProviderOption(provider, key, value)` | `model.ProviderOptions{"openai": openai.ResponsesOptions{...}}` |
| Timeouts | `ai.WithTimeout` | `context.WithTimeout` |
| Middleware | `ai.WrapLanguageModel`, `ai.*Middleware` | `pkg/ai/middleware` |
| Registry | `pkg/provider.Registry` | `ai.NewRegistry`, `ai.NewCustomProvider` |
| Cosine similarity | `ai.CosineSimilarity(a, b) float64` | `ai.CosineSimilarity(vector1, vector2) (float64, error)`, which reports vectors of different lengths as an `InvalidArgumentError` |
| Removed | `ai.SchemaFor`, `ai.GenerateID`, `ai.CreateIDGenerator`, `ai.APIError`, `pkg/provider` | See [Removed APIs](#removed-apis) |

## Providers

Package-level constructors are gone. Create a provider with `New`, then ask it for models. Every provider implements `model.Provider`.

Before:

```go
gpt := openai.Chat("gpt-5.4", openai.WithAPIKey(key))
claude := anthropic.Chat("claude-sonnet-4-5")
reranker := cohere.Rerank("rerank-v3.5")
```

After:

```go
openaiProvider := openai.New(
	openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
)

chatModel := openaiProvider.Chat("gpt-5.4")
responsesModel := openaiProvider.Responses("gpt-5.4")
claude := anthropic.New().Messages("claude-sonnet-4-5")
reranker := cohere.New().Reranking("rerank-v3.5")
```

Notes:

- `openai.Provider.Chat` uses Chat Completions, as `openai.Chat` did. `openai.Provider.LanguageModel` returns the Responses model, which is upstream's default.
- API keys and base URLs fall back to environment variables when a request is sent, not when the provider is created.
- Changed options: Anthropic `WithAPIVersion` was removed and `WithAuthToken` added. OpenRouter `WithHTTPReferer` and `WithSiteURL` became `WithAppURL` and `WithAppName`. All providers gained `WithHeaders`.
- Model methods per provider:

| Package | Methods |
| --- | --- |
| `openai` | `Responses`, `Chat`, `Completion`, `Embedding`, `Image`, `Speech`, `Transcription` |
| `anthropic` | `Messages` (`Chat` is an alias) |
| `google` (new) | `Chat`, `Interactions`, `Embedding`, `Image`, `Speech`, `Transcription`, `Video` |
| `mistral` (new) | `Chat`, `Conversation`, `Embedding`, `Speech`, `Transcription` |
| `cohere` | `Chat`, `Embedding`, `Reranking` |
| `openrouter` | `Chat`, `Embedding` |
| `openaicompatible` (new) | `New(name, baseURL, opts...)`, then `Chat`, `Completion`, `Embedding`, `Image` |

The other 41 provider packages under `pkg/provider` are new. The [README](../README.md#providers) lists them with their environment variables and model kinds.

## Model interfaces

If you implement or wrap a model yourself:

| v0.3.0 | Now |
| --- | --- |
| `model.Model` | `model.LanguageModel` |
| `DoGenerate(ctx, *model.Request) (*model.Response, error)` | `Generate(ctx, model.CallOptions) (*model.GenerateResult, error)` |
| `DoStream(ctx, *model.Request) (*model.StreamResponse, error)` | `Stream(ctx, model.CallOptions) (*model.StreamResult, error)` |
| `model.ResponseFormatSupport` | Removed. Models report unsupported settings as `model.Warning` values. |
| none | `SupportedURLs() map[string][]*regexp.Regexp`: file URLs the provider downloads itself. Other URLs are downloaded by `pkg/ai`. |
| `EmbeddingModel.DoEmbed(ctx, []string)` | `Embed(ctx, model.EmbedOptions)`, plus `MaxEmbeddingsPerCall` and `SupportsParallelCalls` |
| `ImageModel.DoGenerateImage` | `Generate(ctx, model.ImageOptions)`, plus `MaxImagesPerCall` |
| `DoTranscribe`, speech and rerank `Do*` methods | `Transcribe`, `Generate` and `Rerank` with `model.*Options` structs |

`model.CallOptions` replaces `model.Request`:

- `Messages` became `Prompt []model.Message`.
- `MaxTokens` became `MaxOutputTokens`, and `Seed` changed from `*int64` to `*int`.
- `Tools` is now `[]model.Tool` (`model.FunctionTool` or `model.ProviderTool`).
- `Reasoning *ReasoningOptions` became `Reasoning model.ReasoningEffort`.
- `ActiveTools` moved to `ai.GenerateTextOptions`.
- `Modalities`, `Audio` and `Prediction` were removed. OpenAI's prediction moved to `openai.ChatOptions.Prediction`.
- `IncludeRawChunks` was added.

`model.StreamResult` owns the connection. Range over `Parts()` once, or call `Close` without ranging.

Stream chunks were renamed to stream parts. `TextDelta`, `ReasoningDelta`, `ToolInputDelta` and the others now have explicit start and end parts that share an `ID`. `UsageChunk` and `FinishChunk` were merged into `model.Finish`. `ErrorChunk` became `model.StreamError`.

## Calling `GenerateText`

Functional options became fields. Sampling settings are pointers in the embedded `ai.CallSettings`. Instructions and input are in the embedded `ai.Prompt`.

Before:

```go
result, err := ai.GenerateText(ctx, gpt,
	ai.WithSystem("You are a helpful assistant that responds concisely."),
	ai.WithPrompt("Explain concurrency in two sentences."),
	ai.WithTemperature(0.7),
	ai.WithTimeout(30*time.Second),
)
fmt.Println(result.Text, result.Usage.InputTokens)
```

After:

```go
ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
defer cancel()

result, err := ai.GenerateText(ctx, ai.GenerateTextOptions{
	Model: gpt,
	CallSettings: ai.CallSettings{
		Temperature: new(0.7),
	},
	Prompt: ai.Prompt{
		System: "You are a helpful assistant that responds concisely.",
		Prompt: "Explain concurrency in two sentences.",
	},
})
if err != nil {
	return err
}

fmt.Println(result.Text)

if result.Usage.InputTokens != nil {
	fmt.Println(*result.Usage.InputTokens)
}
```

Option mapping:

| v0.3.0 option | Now |
| --- | --- |
| `WithSystem`, `WithPrompt`, `WithMessages` | `Prompt.System`, `Prompt.Prompt`, `Prompt.Messages` |
| `WithTemperature`, `WithTopP`, `WithTopK`, `WithPresencePenalty`, `WithFrequencyPenalty`, `WithSeed`, `WithStopSequences` | `CallSettings` fields with the same names |
| `WithMaxTokens` | `CallSettings.MaxOutputTokens` |
| `WithMaxRetries` | `CallSettings.MaxRetries` (nil means 2, zero disables retries) |
| `WithHeader`, `WithHeaders` | `CallSettings.Headers` |
| `WithTimeout` | `context.WithTimeout` |
| `WithReasoning`, `WithReasoningEffort` | `CallSettings.Reasoning` (`model.ReasoningHigh` and so on). Token budgets are set in provider options. |
| `WithTools`, `WithActiveTools` | `Tools []ai.Tool`, `ActiveTools []string` |
| `WithToolChoice*` | `ToolChoice *model.ToolChoice` |
| `WithMaxSteps(n)` | `StopWhen: []ai.StopCondition{ai.IsStepCount(n)}` |
| `WithStopWhen` | `StopWhen` (see [Steps](#steps-and-callbacks)) |
| `WithPrepareStep` | `PrepareStep` (see [Steps](#steps-and-callbacks)) |
| `WithOnStepFinish`, `WithOnFinish` | `OnStepEnd`, `OnEnd` (upstream's `onStepEnd` and `onEnd`). They no longer return errors. `OnEnd` receives `ai.GenerateTextEndEvent`. |
| `WithProviderOption` | `ProviderOptions` (see [Provider options](#provider-options)) |
| `WithModalities`, `WithAudio`, `WithPrediction` | Removed. Prediction is `openai.ChatOptions.Prediction`. |

`StopWhen` now defaults to one step. A tool loop runs only when you set a stop condition, or when you use `ai.ToolLoopAgent`, which defaults to 20 steps.

System messages inside `Messages` are rejected unless `Prompt.AllowSystemInMessages` is true. Use `Prompt.System` instead.

## Messages and parts

`ai.Message` and its parts were replaced by typed per-role messages in `pkg/model`. Each role accepts only its valid parts.

Before:

```go
ai.Message{
	Role: ai.RoleUser,
	Parts: []ai.Part{
		ai.TextPart{Text: "What is in this image?"},
		ai.ImagePart{URL: "https://example.com/cat.png"},
	},
}

ai.Message{
	Role: ai.RoleTool,
	Parts: []ai.Part{
		ai.ToolResultPart{ToolCallID: "call_1", ToolName: "get_weather", Content: "72F and sunny"},
	},
}
```

After:

```go
[]model.Message{
	model.UserMessage{
		Content: []model.UserPart{
			model.TextPart{
				Text: "What is in this image?",
			},
			model.FilePart{
				MediaType: "image/*",
				Data: model.FileURL{
					URL: imageURL,
				},
			},
		},
	},
	model.AssistantMessage{
		Content: []model.AssistantPart{
			model.ToolCallPart{
				ToolCallID: "call_1",
				ToolName:   "get_weather",
				Input:      json.RawMessage(`{"city":"Tokyo"}`),
			},
		},
	},
	model.ToolMessage{
		Content: []model.ToolPart{
			model.ToolResultPart{
				ToolCallID: "call_1",
				ToolName:   "get_weather",
				Output: model.TextOutput{
					Value: "72F and sunny",
				},
			},
		},
	},
}
```

| v0.3.0 | Now |
| --- | --- |
| `Message.Content` string | `model.SystemMessage{Content: "..."}`, or a `model.TextPart` for other roles |
| `ImagePart{URL, Data, MIMEType}` | `FilePart{MediaType: "image/...", Data: model.FileURL{...}}` or `model.FileBytes{Data: ...}` |
| `FilePart{URL, Data, MIMEType, Filename}` | `FilePart{Filename, MediaType, Data}` (`Data` is `FileBytes`, `FileURL`, `FileReference` or `FileText`) |
| `ReasoningPart{Signature, ID, Format, Redacted}` | `ReasoningPart{Text, ProviderOptions}`. Signatures and encrypted content are kept in provider options. |
| `ToolCallPart.Args` | `ToolCallPart.Input` |
| `ToolResultPart{Content, IsError}` | `ToolResultPart{Output}`, where `Output` is `TextOutput`, `JSONOutput`, `ErrorTextOutput`, `ErrorJSONOutput`, `ExecutionDeniedOutput` or `ContentOutput` |

To continue a conversation, append `result.ResponseMessages` to the history:

```go
result, err := ai.GenerateText(ctx, ai.GenerateTextOptions{
	Model: gpt,
	Prompt: ai.Prompt{
		Messages: history,
	},
})
if err != nil {
	return nil, err
}

return append(history, result.ResponseMessages...), nil
```

## Tools

`ai.NewTool` generates the input schema from a Go type, replacing `ai.SchemaFor`, and decodes the input before calling `Execute`. The same `json` and `jsonschema` struct tags work.

Before:

```go
weather := ai.Tool{
	Name:        "get_weather",
	Description: "Get the current weather for a city.",
	InputSchema: ai.SchemaFor[WeatherInput](),
	Execute: func(ctx context.Context, input json.RawMessage) (string, error) {
		var params WeatherInput
		if err := json.Unmarshal(input, &params); err != nil {
			return "", err
		}
		return "72F and sunny in " + params.City, nil
	},
}

result, err := ai.GenerateText(ctx, gpt,
	ai.WithPrompt("What's the weather in Tokyo?"),
	ai.WithTools(weather),
	ai.WithMaxSteps(3),
)
```

After:

```go
weather := ai.NewTool(ai.ToolDefinition[WeatherInput, string]{
	Name:        "get_weather",
	Description: "Get the current weather for a city.",
	Execute: func(ctx context.Context, input WeatherInput, options ai.ToolExecutionOptions) (string, error) {
		return "72F and sunny in " + input.City, nil
	},
})

result, err := ai.GenerateText(ctx, ai.GenerateTextOptions{
	Model: gpt,
	Prompt: ai.Prompt{
		Prompt: "What's the weather in Tokyo?",
	},
	Tools: []ai.Tool{
		weather,
	},
	StopWhen: []ai.StopCondition{
		ai.IsStepCount(3),
	},
})
```

A tool literal with a hand-written schema still works. `Execute` now takes `ai.ToolExecutionOptions` and returns `any`. Strings are sent as text and other values as JSON, unless you set `ToModelOutput`.

```go
ai.Tool{
	Name:        "calculate",
	Description: "Evaluate a math expression.",
	InputSchema: json.RawMessage(`{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}`),
	Execute: func(ctx context.Context, input json.RawMessage, options ai.ToolExecutionOptions) (any, error) {
		return "4", nil
	},
}
```

Other changes:

- `Tool.Strict` changed from `bool` to `*bool`.
- A failed `Execute` is sent to the model as an error result. It does not end the generation.
- `ToolResult.Content string` and `IsError` became `ToolResult.Output any`. Failed calls are reported as `ai.ToolError` content.
- Tool choice: `WithToolChoiceNamed("get_weather")` becomes:

  ```go
  &model.ToolChoice{
  	Type:     model.ToolChoiceTool,
  	ToolName: "get_weather",
  }
  ```

- Provider tools are new. For example: `ai.ProviderExecutedTool(anthropic.WebSearch20250305(anthropic.WebSearchArgs{}))`.
- New tool features: `NeedsApproval` and `ToolApproval` (an `ai.ToolApprovalFunc` or a per-tool `ai.ToolApprovals` map, with optional `ToolApprovalSecret` signing), `Tool.OutputSchema` (generated by `ai.NewTool` from the output type), `RepairToolCall`, `RefineToolInput`, `ToolCallConcurrency`, `ToolOrder`, the streaming input callbacks, and preliminary results through `ToolExecutionOptions.Preliminary`. `ai.ExecuteTool` runs a tool outside a generation and reports its preliminary outputs.

## Steps and callbacks

```go
StopWhen: []ai.StopCondition{
	func(ctx context.Context, event ai.StopConditionEvent) bool {
		return len(event.Steps) >= 5
	},
	ai.HasToolCall("final_answer"),
},
```

`PrepareStep` no longer changes a `*model.Request` in place. It returns overrides, and a nil result keeps the current settings:

```go
PrepareStep: func(ctx context.Context, options ai.PrepareStepOptions) (*ai.PrepareStepResult, error) {
	if options.StepNumber == 0 {
		return &ai.PrepareStepResult{
			ToolChoice: &model.ToolChoice{
				Type: model.ToolChoiceRequired,
			},
		}, nil
	}

	return nil, nil
},
```

`ai.Step` became `ai.StepResult`. Its text, tool calls and files are methods (`step.Text()`, `step.ToolCalls()`) derived from the ordered `Content`.

## Results

| v0.3.0 `GenerateResult` | Now `GenerateTextResult` |
| --- | --- |
| `Text`, `Files`, `Sources`, `ToolCalls`, `ToolResults`, `Steps` | Same names. Tool calls and results gain static and dynamic variants. |
| `Reasoning []ai.Reasoning`, `ReasoningText()` | `Reasoning []ai.ContentPart`, `ReasoningText string` |
| `FinishReason model.FinishReason` | `FinishReason model.UnifiedFinishReason`, plus `RawFinishReason` |
| `Usage` (`int` counts) | `Usage ai.LanguageModelUsage` (`*int` counts, nil when not reported), with `InputTokenDetails` and `OutputTokenDetails` |
| `Metadata model.ProviderMetadata` (struct) | `ProviderMetadata` (map keyed by provider), plus `Request` and `Response` |
| `Warnings []string` | `Warnings []model.Warning` |
| none | `Content`, `ResponseMessages`, `FinalStep`, `Output` |

## Streaming

`StreamTextResult` has a single consumer. Range over exactly one stream once, then call `Result`. Breaking out of the loop stops generation. Always `Close` the result.

Before:

```go
stream, err := ai.StreamText(ctx, gpt, ai.WithPrompt("Write a short poem about Go."))
for delta := range stream.Text() {
	fmt.Print(delta)
}
final, err := stream.Result()
```

After:

```go
stream, err := ai.StreamText(ctx, ai.StreamTextOptions{
	Model: gpt,
	Prompt: ai.Prompt{
		Prompt: "Write a short poem about Go.",
	},
})
if err != nil {
	return err
}
defer func() {
	err = errors.Join(err, stream.Close())
}()

for delta := range stream.TextStream() {
	fmt.Print(delta)
}

result, err := stream.Result()
```

| v0.3.0 `StreamResult` | Now `StreamTextResult` |
| --- | --- |
| `Text()` | `TextStream()` |
| `Chunks()` (`model.Chunk`) | `FullStream()` (`ai.StreamPart`: `TextDeltaPart`, `ReasoningDeltaPart`, `ToolCall`, `ToolResult`, `ErrorPart`, `FinishPart` and others) |
| `Reasoning()`, `Files()`, `Sources()` | Read these from `FullStream()`, or from `Result()` after the stream ends |
| `Metadata()` | `Result()` (`ProviderMetadata`, `Response`) |
| `Result()` | `Result()` returns `*GenerateTextResult` |

```go
for part := range stream.FullStream() {
	switch part := part.(type) {
	case ai.TextDeltaPart:
		fmt.Print(part.Text)
	case ai.ReasoningDeltaPart:
		fmt.Print(part.Text)
	case ai.ErrorPart:
		fmt.Println(part.Err)
	}
}
```

New streaming options: `Transforms` (for example `ai.SmoothStream`), `OnChunk`, `OnError`, `OnAbort`, and `Include.RawChunks`. To serve a `useChat` frontend, use `pkg/ai/ui` (`ToMessageStream`, `PipeMessageStreamToResponse`).

## Structured output

`GenerateObject` and `StreamObject` were removed, following upstream's deprecation. Set `Output` on `GenerateText` or `StreamText` instead.

Before:

```go
object, err := ai.GenerateObject[Recipe](ctx, gpt, ai.WithPrompt("Generate a lasagna recipe."))
fmt.Println(object.Object.Name)
```

After:

```go
result, err := ai.GenerateText(ctx, ai.GenerateTextOptions{
	Model: gpt,
	Prompt: ai.Prompt{
		Prompt: "Generate a lasagna recipe.",
	},
	Output: ai.ObjectOutput[Recipe](ai.ObjectOutputOptions{}),
})
if err != nil {
	return err
}

recipe, err := ai.OutputAs[Recipe](result.Output)
```

Streaming partial objects replaces `ObjectStreamResult.JSON()`. Each partial value is a `Recipe`:

```go
for partial := range stream.PartialOutputStream() {
	recipe := partial.(Recipe)
	fmt.Println(len(recipe.Ingredients))
}

output, err := stream.Output()
```

Other output modes: `ai.ArrayOutput[T]` (use `ElementStream` for completed elements), `ai.ChoiceOutput` and `ai.JSONOutput`. `ObjectResult.RawJSON` has no direct replacement. The raw text is in `result.Text`.

## Provider options

Provider options are typed structs, keyed by provider namespace. Maps and raw JSON are also accepted.

Before:

```go
ai.WithProviderOption("openai", "store", false)
```

After:

```go
ProviderOptions: model.ProviderOptions{
	"openai": openai.ResponsesOptions{
		Store: new(false),
	},
},
```

Each provider documents its option types, for example `openai.ChatOptions`, `openai.ResponsesOptions`, `anthropic.MessagesOptions` and `google.ChatOptions`. Reasoning budgets that used to go through `WithReasoning` are also set here:

```go
model.ProviderOptions{
	"anthropic": anthropic.MessagesOptions{
		Thinking: &anthropic.ThinkingOptions{
			Type:         anthropic.ThinkingEnabled,
			BudgetTokens: new(4096),
		},
	},
}
```

Provider options can also be set on messages, parts and tools. Provider metadata from results (`model.ProviderMetadata`) uses the same form. Read either with `model.ProviderValue[T]`.

## Embeddings and media

Each function takes one options struct. The model is a field, and results include warnings, provider metadata and response information.

Before:

```go
single, _ := ai.Embed(ctx, embedder, "sunny day at the beach")
many, _ := ai.EmbedMany(ctx, embedder, []string{"sunny day", "rainy night"})

image, _ := ai.GenerateImage(ctx, openai.Image("gpt-image-1"), model.ImageRequest{Prompt: "A watercolor fox", Count: 2, Size: "1024x1024"})
speech, _ := ai.GenerateSpeech(ctx, openai.Speech("gpt-4o-mini-tts"), model.SpeechRequest{Input: "Hello there", Voice: "alloy"})
transcript, _ := ai.Transcribe(ctx, openai.Transcription("whisper-1"), model.TranscriptionRequest{Audio: model.AudioInput{Data: audio, MIMEType: "audio/mpeg"}})
ranking, _ := ai.Rerank(ctx, cohere.Rerank("rerank-v3.5"), model.RerankRequest{Query: "go concurrency", Documents: []string{"goroutines", "promises"}})
```

After:

```go
single, err := ai.Embed(ctx, ai.EmbedOptions{
	Model: embedder,
	Value: "sunny day at the beach",
})

many, err := ai.EmbedMany(ctx, ai.EmbedManyOptions{
	Model: embedder,
	Values: []string{
		"sunny day",
		"rainy night",
	},
})

image, err := ai.GenerateImage(ctx, ai.GenerateImageOptions{
	Model:  provider.Image("gpt-image-1"),
	Prompt: "A watercolor fox",
	N:      2,
	Size:   "1024x1024",
})

speech, err := ai.GenerateSpeech(ctx, ai.GenerateSpeechOptions{
	Model: provider.Speech("gpt-4o-mini-tts"),
	Text:  "Hello there",
	Voice: "alloy",
})

transcript, err := ai.Transcribe(ctx, ai.TranscribeOptions{
	Model: provider.Transcription("whisper-1"),
	Audio: model.FileBytes{
		Data: audio,
	},
})

ranking, err := ai.Rerank(ctx, ai.RerankOptions[string]{
	Model:     cohere.New().Reranking("rerank-v3.5"),
	Query:     "go concurrency",
	Documents: []string{"goroutines", "promises"},
	TopN:      new(1),
})
```

| v0.3.0 | Now |
| --- | --- |
| `EmbedResult.Usage` | `*model.EmbeddingUsage` (nil when not reported) |
| `ImageRequest.Count` | `GenerateImageOptions.N`. Calls are split by `MaxImagesPerCall`. |
| `ImageRequest.Images []ImageInput` | `Images []model.FileData`, plus `Mask` |
| `ImageRequest.Quality`, `Background`, `OutputFormat`, `OutputCompress` | OpenAI provider options (`openai.ImageOptions`) |
| `ImageResponse.Images` (`URL`, `Data`, `RevisedPrompt`) | `GenerateImageResult.Images []ai.GeneratedFile` (`Data`, `MediaType`). Revised prompts are in provider metadata. |
| `SpeechRequest.Input`, `ResponseFormat` | `Text`, `OutputFormat` |
| `SpeechResponse.Audio []byte` | `GenerateSpeechResult.Audio` (`GeneratedAudioFile`: `Data`, `MediaType`, `Format`) |
| `TranscriptionRequest.Audio AudioInput` | `TranscribeOptions.Audio model.FileData`. URLs are downloaded, and the media type is detected. |
| `TranscriptionRequest.Language`, `Prompt`, `Temperature` | Provider options (for example `openai.TranscriptionOptions`) |
| `TranscriptionResponse.DurationSeconds` | `TranscribeResult.Duration time.Duration` |
| `RerankRequest.Documents []string` | `RerankOptions[Document].Documents`: any JSON-object type or `string` |
| `RerankRequest.MaxTokensPerDoc` | `cohere.RerankingOptions.MaxTokensPerDoc` |
| `RerankResponse.Results` (`Index`, `Score`, `Document`) | `RerankResult.Ranking` (`OriginalIndex`, `Score`, `Document`), plus `RerankedDocuments` |

## Middleware

Middleware moved to `pkg/ai/middleware`. A middleware is now a struct of optional hooks (`TransformParams`, `WrapGenerate`, `WrapStream`) instead of a `Wrap(model.Model) model.Model` interface.

Before:

```go
wrapped := ai.WrapLanguageModel(gpt, ai.ExtractReasoningMiddleware(), ai.DefaultSettingsMiddleware(model.Request{}))
```

After:

```go
extractReasoning, err := middleware.ExtractReasoning(middleware.ExtractReasoningOptions{
	TagName: "think",
})
if err != nil {
	return nil, err
}

wrapped := middleware.WrapLanguageModel(middleware.WrapLanguageModelOptions{
	Model: gpt,
	Middleware: []middleware.Language{
		extractReasoning,
		middleware.DefaultSettings(model.CallOptions{
			Temperature: new(0.2),
		}),
	},
})
```

| v0.3.0 | Now |
| --- | --- |
| `ai.DefaultSettingsMiddleware(model.Request)` | `middleware.DefaultSettings(model.CallOptions)` |
| `ai.ExtractJSONMiddleware()` | `middleware.ExtractJSON(middleware.ExtractJSONOptions{})` |
| `ai.ExtractReasoningMiddleware()` | `middleware.ExtractReasoning(options)`. Set `TagName`, or both `OpeningTag` and `ClosingTag`. It returns an error when no delimiters are set. |
| `ai.SimulateStreamingMiddleware(delay)` | `middleware.SimulateStreaming()`. It has no delay; use `ai.SmoothStream` for pacing. |
| none | `DefaultInstructions`, `AddToolInputExamples`, `DefaultEmbeddingSettings`, `WrapEmbeddingModel`, `WrapImageModel`, `WrapProvider` |

## Registry

`pkg/provider` (`provider.Provider`, `provider.Registry`) was removed. Providers implement `model.Provider`. The registry resolves combined `provider:model` IDs for every model kind.

Before:

```go
registry := provider.NewRegistry()
registry.Register("openai", openai.New())
p, _ := registry.Get("openai")
gpt := p.Chat("gpt-5.4")
```

After:

```go
registry := ai.NewRegistry(ai.RegistryOptions{
	Providers: map[string]model.Provider{
		"openai":    openai.New(),
		"anthropic": anthropic.New(),
	},
})

claude, err := registry.LanguageModel("anthropic:claude-sonnet-4-5")
```

`RegistryOptions` also accepts `Separator`, `LanguageModelMiddleware` and `ImageModelMiddleware`. Use `ai.NewCustomProvider` for aliases and fixed model sets.

## Errors

| v0.3.0 | Now |
| --- | --- |
| `ai.APIError`, `ai.NewAPIError` | `*model.APICallError` (use `errors.As`) |
| `ai.ErrNoContent` and other `ai` sentinels | Sentinels in `pkg/model` (`ErrUnsupported`, `ErrNoSuchModel`, `ErrLoadAPIKey`, `ErrInvalidResponse` and others), plus typed errors in `pkg/ai` (`NoOutputGeneratedError`, `RetryError`, `NoSuchToolError`, `InvalidToolInputError` and others) |

```go
var apiError *model.APICallError
if errors.As(err, &apiError) {
	fmt.Println(apiError.StatusCode)
}

if errors.Is(err, model.ErrUnsupported) {
	fmt.Println("unsupported")
}
```

## Removed APIs

| Removed | Replacement |
| --- | --- |
| `openai.Chat`, `openai.Embedding` and the other package-level constructors | `openai.New().Chat(...)` and the other provider methods |
| `openai.DecodeResponse`, `DecodeStream`, `MarshalRequest`, `DoStreamRequest`, `ClientConfig` (OpenAI and Anthropic) | None. Wire formats are internal. |
| `ai.SchemaFor` | `ai.NewTool` and `ai.ObjectOutput` generate schemas, or pass `json.RawMessage`. |
| `ai.GenerateID`, `ai.CreateIDGenerator` | None. Providers that generate IDs accept `WithIDGenerator`, and `pkg/ai/ui` accepts ID generator hooks. |
| `ai.GenerateObject`, `ai.StreamObject` | `Output` on `GenerateText` or `StreamText` |
| `pkg/provider` | `model.Provider`, `ai.NewRegistry` |
| `ai.Usage` alias, `model.Usage.TotalTokens()` | `ai.LanguageModelUsage.TotalTokens` field |

## Changes since the first rewrite snapshots

These changes break code written against early, untagged snapshots of the rewrite. Code migrating straight from v0.3.0 only needs the current API shown here.

### Telemetry uses optional per-operation interfaces

`ai.Telemetry` now has only `OnError`, which every operation calls. Each operation reports its lifecycle to its own optional interface, found with a type assertion: `ai.TextTelemetry` (GenerateText, StreamText and agents), `ai.EmbedTelemetry`, `ai.RerankTelemetry`, `ai.SpeechTelemetry`, `ai.TranscriptionTelemetry`, `ai.TranslationTelemetry` (new; `StreamTranslateOptions.Telemetry`) and `ai.DecisionTelemetry`. Implement only the interfaces you record, or embed `ai.NopTelemetry` to implement all of them and override some methods.

The text generation events are renamed:

| Before | Now |
| --- | --- |
| `OnStart` | `OnGenerateTextStart` |
| `OnStepStart` | `OnGenerateTextStepStart` |
| `OnStepEnd` | `OnGenerateTextStepEnd` |
| `OnEnd` | `OnGenerateTextEnd` |
| `OnAbort` | `OnGenerateTextAbort` |
| `ai.GenerateTextFinishEvent` | `ai.GenerateTextEndEvent` |

A method with an old name on a type that embeds `ai.NopTelemetry` still compiles but is no longer called; rename it.

```go
type usageLogger struct {
	ai.NopTelemetry
}

func (usageLogger) OnGenerateTextEnd(ctx context.Context, event ai.TelemetryEvent[ai.GenerateTextEndEvent]) {
	slog.InfoContext(ctx, "generation finished", "function", event.FunctionID)
}
```

Pass integrations per call in `Telemetry: ai.TelemetryOptions{Integrations: ...}`.

### Generation callbacks end with `OnStepEnd` and `OnEnd`

`GenerateTextOptions`, `StreamTextOptions`, `ai.ToolLoopAgentSettings`, `ai.AgentCallOptions`, `ui.CreateAgentStreamOptions` and the harness agent settings rename `OnStepFinish` to `OnStepEnd` and `OnFinish` to `OnEnd`, matching upstream and the `OnEnd` callbacks of the other operations.

### Batch, upload and realtime option changes

- The batch options name the batch interface `API`, like `UploadFileOptions.API` and `UploadSkillOptions.API`: `ai.StartBatchOptions{API: provider.Batch(), ...}`.
- `ai.GetBatchStatus` returns `*ai.Batch`, the reference with its latest status, like `StartBatch` and `ListBatches`.
- `UploadSkillOptions` gains `Headers` (also on `model.UploadSkillOptions`), and both upload options gain `LogWarnings`.
- `ai.RealtimeRelay.Header` is now `Headers`.
- `ai.TranslateResult.Response` and `ai.NoTranslationGeneratedError.Response` are now `Responses []model.ResponseInfo`, like transcription.
- Tool approval decisions with an unknown `ToolApprovalStatusType` fail with `*ai.InvalidArgumentError` instead of being treated as decided.
- `ToolCallers` without tools, and tool callers without their `Bind` or `PrepareProviderOptions` function, fail with `*ai.InvalidArgumentError`.
- `ui.ChatStream.Chunks` can be ranged over once; a second iteration yields `model.ErrStreamConsumed`.
- A tool that fails because its `TimeoutConfiguration` limit elapsed reports a `*ai.TimeoutError`.

### Realtime client events serialize to frames

`model.RealtimeModel.SerializeClientEvent` returns `(*model.RealtimeFrame, error)`. A frame is a text or binary WebSocket message, so codecs can send JSON, plain commands or raw audio. Build frames with `model.NewRealtimeJSONFrame`, `model.NewRealtimeTextFrame` or `model.NewRealtimeBinaryFrame`. A nil frame sends nothing.

```go
func serializeClientEvent(event model.RealtimeClientEvent) (*model.RealtimeFrame, error) {
	message, err := json.Marshal(event)
	if err != nil {
		return nil, err
	}

	return model.NewRealtimeJSONFrame(message), nil
}
```

### DeepSeek files and Alibaba video use the shared contracts

`deepseek.Provider.Files` returns a `model.Files`, and `alibaba.Provider.Video` returns a `model.VideoModel`. Use them with `ai.UploadFile` and `ai.GenerateVideo` like any other provider:

```go
uploaded, err := ai.UploadFile(ctx, ai.UploadFileOptions{
	API: deepseek.New().Files(),
	Data: model.FileBytes{
		Data: data,
	},
	MediaType: "image/png",
})
if err != nil {
	return err
}

video, err := ai.GenerateVideo(ctx, ai.GenerateVideoOptions{
	VideoSettings: ai.VideoSettings{
		Model:  alibaba.New().Video("wan2.6-t2v"),
		Prompt: "A paper boat drifting down a rainy street.",
	},
})
```

### Provider renames

- **Google:** the Gemini language model type is `google.ChatModel`, returned by `Provider.Chat`. The grounding tools are `google.Search(google.SearchArgs{...})`, `google.Maps()` and `google.EnterpriseWebSearch()`. `Provider.Image(modelID)` takes no settings; set `ai.GenerateImageOptions.MaxImagesPerCall` to request more than one image per call, and pass `google.ImageOptions` in provider options.
- **Mistral:** `mistral.WebSearch()` and `mistral.WebSearchPremium()` are functions that return a `model.ProviderTool`.
- **OpenAI:** `openai.FileSearchFilter.Value` is a `*openai.FileSearchValue`. Create it with `openai.FileSearchString`, `FileSearchNumber`, `FileSearchBool` or `FileSearchStrings`.
- **Model kinds:** the type naming model kinds in errors and registries is `model.Kind`, with the constants `model.LanguageModelKind`, `model.VideoModelKind` and so on.
- **OpenAI-compatible and DeepSeek configs:** `ChatConfig`, `CompletionConfig`, `EmbeddingConfig` and `ImageConfig` in `openaicompatible`, and `deepseek.ChatConfig`, have only `BuildURL`. Wrap a URL function that cannot fail with `openaicompatible.InfallibleURL`.
- **Prodia:** the language model is `prodia.ChatModel`, returned by `Provider.Chat`.
- **HeyGen:** the completed video metadata is `heygen.VideoMetadata`, and each video in it is a `heygen.GeneratedVideoMetadata`, matching KlingAI and Replicate.
- **Cartesia and ElevenLabs:** the streaming options are `cartesia.TranscriptionStreamingOptions` and `elevenlabs.TranscriptionStreamingOptions`, matching xAI.
- **AssemblyAI:** `assemblyai.WithPollingInterval` is `assemblyai.WithPollInterval`.
- **Async job errors:** failures and poll timeouts wrap exported sentinels named `Err<Kind>Failed` and `Err<Kind>Timeout`. `gladia.ErrPollingTimedOut` and `revai.ErrPollingTimedOut` are `ErrTranscriptionTimeout`. New sentinels: `blackforestlabs.ErrImageGenerationFailed` and `ErrImageGenerationTimeout`, `replicate.ErrImageGenerationFailed` and `ErrImageGenerationTimeout`, `luma.ErrImageGenerationFailed` and `ErrImageGenerationTimeout`, `fal.ErrTranscriptionTimeout`, `fireworks.ErrImageGenerationTimeout` (polling timeouts previously wrapped `fireworks.ErrImageGenerationFailed`), and `topaz.ErrImageEnhancementFailed` and `ErrImageEnhancementTimeout`.
- **KlingAI and TypeSafe:** KlingAI credentials are trimmed, and a blank credential counts as unset. `typesafeai.New` reads `TYPESAFE_AI_BASE_URL` when the provider is created, like the other providers.
- **Authentication hooks:** `openai.WithAuthHeaders`, `anthropic.WithAuthHeaders` and `google.WithAuthHeaders` take an `AuthHeadersFunc`, `func(ctx context.Context) (http.Header, error)`. They replace `openai.WithAuthHeaders(func() (http.Header, error))`, `anthropic.WithHeadersFunc` and `google.WithAuthentication` with `google.Authenticator`.
- **URL hooks:** `openai.WithURLBuilder` and `openai.URLBuilder` are `openai.WithRequestURL` and `openai.RequestURLFunc`. See [provider hook names](#provider-hook-names).
- **OpenAI options namespace:** Responses models read the namespace set with `openai.WithOptionsNamespace`, then `"openai"`. Providers named after Azure no longer switch to `"azure"` on their own; `azure` sets it. Batch options are read only from `"openai"`.
- **OpenAI realtime:** `Provider.Realtime` returns the Realtime API `*openai.RealtimeModel`, and `Provider.Live` returns the Live API `*openai.LiveModel`. They replace `RealtimeAPI` and `LiveAPI`. `Provider.RealtimeModel` still picks the Live API for `gpt-live-1`.
- **Decision models:** `anthropic.Provider.Decision` and `google.Provider.Decision` return `*anthropic.DecisionModel` and `*google.DecisionModel`. `DecisionModel(modelID)` still returns a `model.DecisionModel`.
- **Namespaces:** `openai.Namespace`, `anthropic.Namespace` and `google.Namespace` name each provider's options and metadata namespace.
- **Token sources:** `azure.TokenProvider` and `azure.WithTokenProvider` are `azure.TokenSource` and `azure.WithTokenSource`. The `googlevertex/maas`, `googlevertex/xai` and `googlevertex/anthropic` `TokenSource` types are aliases of `googlevertex.TokenSource`.
- **Credentials only reach their own origin:** Azure token sources, Amazon Bedrock Anthropic and Mantle clients and `anthropicaws` add credentials only to requests for the provider's configured endpoints.
- **Google Vertex streaming transcription:** the live metadata is `googlevertex.GeminiTranscriptionStreamMetadata`, an alias of `google.TranscriptionStreamMetadata`, still stored under `"google"`.
- **Google batch and video IDs:** batch IDs must have the form `batches/{id}` and video operation names `models/{model}/operations/{id}`; other values return `model.ErrInvalidArgument`. Polling a background interaction past its timeout cancels it and wraps `google.ErrInteractionPollTimeout`.
- **More options:** `azure`, `googlevertex`, `googlevertex/maas` and `googlevertex/xai` accept `WithName`, and `amazonbedrock/anthropic` accepts `WithIDGenerator`.

```go
tools := []ai.Tool{
	ai.ProviderExecutedTool(google.Search(google.SearchArgs{})),
	ai.ProviderExecutedTool(google.Maps()),
	ai.ProviderExecutedTool(mistral.WebSearch()),
}

fileSearch := openai.FileSearch(openai.FileSearchArgs{
	VectorStoreIDs: []string{
		"vs_123",
	},
	Filters: &openai.FileSearchFilter{
		Type:  openai.FileSearchIn,
		Key:   "category",
		Value: openai.FileSearchStrings("guides", "faq"),
	},
})

err := model.NoSuchModelError("example", model.VideoModelKind, "video-1")
```

### Provider hook names

Providers that other providers build on name their hooks the same way:

- `With…BaseURLResolver` takes a function that returns a base URL, to which the provider adds the API path.
- `With…URL`, such as `WithRequestURL`, takes a function that returns the full URL of one request.
- `WithAuthHeaders` takes an `AuthHeadersFunc` that returns the authentication headers of one request.
- `TokenSource` types return an OAuth access token, and `WithTokenSource` sets one.

## New packages

These packages have no v0.3.0 equivalent.

| Package | Contents |
| --- | --- |
| `pkg/provider/...` | 41 new providers, from `alibaba` to `zai`, each with `New` and typed model constructors |
| `pkg/mcp` | Model Context Protocol client: server tools as `[]ai.Tool`, resources, prompts, elicitation, Streamable HTTP, SSE and stdio transports, OAuth |
| `pkg/harness` | Drives coding-agent runtimes in a sandbox. Adapters: `acp`, `claudecode`, `codex`, `cursor`, `deepagents`, `fx`, `githubcopilot`, `grokbuild`, `opencode` |
| `pkg/sandbox` | The sandbox session contract, with `sandbox/vercel` (Vercel Sandbox) and `sandbox/local` (an unisolated local shell) |
| `pkg/tools/bash` | A bash tool over any sandbox session |
| `pkg/devtools` | Records calls for local debugging and serves a viewer (`pkg/devtools/cmd/devtools`) |
| `pkg/policyopa` | Open Policy Agent tool approval policies and capability middleware |
| `github.com/aarock1234/ai/otel` | OpenTelemetry integration (nested module) |
| `github.com/aarock1234/ai/codemode` | Lets models write TypeScript that calls your tools (nested module) |
| `github.com/aarock1234/ai/tui` | Runs an `ai.Agent` in a terminal UI (nested module) |

`pkg/ai` also gained `GenerateVideo`, `StartVideo` and `GetVideoStatus`, `UploadFile` and `UploadSkill`, `StreamTranscribe`, `StreamTranslate`, `ConnectRealtime`, `Decide`, the batch functions (`StartBatch` and others), tool search and tool callers, and per-call telemetry and warning logging. See [feature parity](feature-parity.md) for the full list.
