# Feature parity

This SDK is a Go port of the Vercel AI SDK. This page maps every upstream package and module to its Go location and status. It is based on the current source and records what the code does, not runtime conformance: nothing has been checked against live providers or a test suite.

## Pinned baseline

| Item | Value |
| --- | --- |
| Core package | `ai@7.0.146` |
| Provider contract | `@ai-sdk/provider@4.0.31` (v4 model specifications) |
| Source commit | [`a1db2092f9cbdf20b35c7d374c79a0aeceefef2a`](https://github.com/vercel/ai/tree/a1db2092f9cbdf20b35c7d374c79a0aeceefef2a) |
| Go modules | `github.com/aarock1234/ai` (core), plus the nested modules `github.com/aarock1234/ai/otel`, `github.com/aarock1234/ai/codemode` and `github.com/aarock1234/ai/tui`; Go `1.27.3` |

Status values:

- **Present**: the Go code covers the upstream package or module. Deliberate Go adaptations are not gaps.
- **Partial**: part of it is ported. The missing pieces are listed.
- **Not applicable**: there is no Go equivalent, and none is planned. The reason is given.
- **Deferred**: not ported yet, pending an owner decision. See the [roadmap](roadmap.md).

## Go adaptations

These are deliberate differences from the TypeScript API. They are not gaps.

- **Cancellation and deadlines** come from the `context.Context` passed to every call. There is no `abortSignal` option, and the total timeout (upstream's numeric `timeout` or `totalMs`) is a context deadline. Step, first-chunk, chunk and tool timeouts are `ai.TimeoutConfiguration` durations.
- **One options struct per function**: `GenerateText(ctx, GenerateTextOptions)`, `Embed(ctx, EmbedOptions)` and so on. Callbacks are `func` fields that receive the context and one event struct.
- **Pull-based, single-consumer streams.** `model.StreamResult.Parts` and the `StreamTextResult` streams are `iter.Seq` values that can be ranged over once. Generation runs on the consumer's goroutine, and breaking out of the loop releases the HTTP connection. There is no tee: use exactly one of `FullStream`, `TextStream`, `PartialOutputStream` or `ElementStream`, then call `Result`.
- **Typed messages per role.** `model.SystemMessage`, `UserMessage`, `AssistantMessage` and `ToolMessage` each accept only the part types the specification allows. Tagged unions are sealed interfaces.
- **Images are `model.FilePart`** with an `image/*` media type. File payloads are `FileBytes`, `FileURL`, `FileReference` (provider file IDs) or `FileText`, never base64 strings.
- **Provider options** are `model.ProviderOptions` (`map[string]any`) keyed by provider namespace. A value can be the provider's typed options struct (for example `openai.ResponsesOptions`), a map or raw JSON. Providers read them with `model.ProviderValue[T]`. `ProviderMetadata` uses the same form.
- **Tools** are an ordered `[]ai.Tool`, not a name-keyed object. `ai.NewTool[Input, Output]` generates the input schema from a Go type (`json` and `jsonschema` struct tags) and decodes inputs before calling the typed function. Zod, Valibot and Standard Schema are replaced by Go types plus `internal/schema`.
- **Tool and runtime context** travel as `context.Context` values. This includes upstream's `experimental_sandbox` tool option: attach a session with `sandbox.WithSession` and read it in a tool with `sandbox.SessionFromContext`.
- **Typed output** is returned as `any`. `ai.OutputAs[T]` converts it.
- **Optional values** are pointers (`*int`, `*float64`, `*bool`). Upstream booleans that default to true become `*bool`.
- **Errors** are typed values that wrap their causes, plus sentinel errors in `pkg/model` (`ErrUnsupported`, `ErrNoSuchModel` and others). Inspect them with `errors.Is` and `errors.As`.
- **Durations** are `time.Duration` instead of millisecond numbers.
- **One contract version.** Only v4 models exist, so there are no v2/v3 adapters.
- **Optional capabilities are optional interfaces.** Files, skills, batch, video, decision, realtime and speech translation support is detected with interfaces such as `model.FilesProvider` and `model.VideoOperator`, so providers without them are unaffected.
- **No global state.** There is no global default provider, no string model IDs outside a `Registry`, no default provider instances (such as upstream's `openai` or `claudeCode` exports; call `New`), no user-agent suffix, no telemetry registry and no default warning logger. Telemetry integrations and the warning logger (`LogWarnings`, for example `ai.SlogWarnings`) are per-call options. Warnings are always returned on results.
- **Nested modules isolate dependencies.** `otel/` (OpenTelemetry), `codemode/` (the goja JavaScript interpreter and the esbuild TypeScript transformer) and `tui/` (Bubble Tea and Lip Gloss) are separate modules, so the core module depends only on a JSON Schema validator.

## Upstream packages

Every directory in upstream `packages/` has one row.

| Upstream package | Status | Go location | Notes |
| --- | --- | --- | --- |
| `ai` | Partial | `pkg/ai`, `pkg/ai/middleware`, `pkg/ai/ui` | See [Core modules](#core-modules-packagesaisrc). Gaps: the mock models in `test` (deferred with tests). |
| `provider` | Present | `pkg/model` | See [Provider contracts](#provider-contracts-packagesprovidersrc). |
| `provider-utils` | Present | `internal/*`, `pkg/model`, `pkg/sandbox` | See [Provider utilities](#provider-utilities-packagesprovider-utilssrc). |
| `alibaba` | Present | `pkg/provider/alibaba` | Qwen chat (thinking, explicit context caching), embeddings and Wan video (`model.VideoModel`). |
| `amazon-bedrock` | Present | `pkg/provider/amazonbedrock`, `amazonbedrock/anthropic`, `amazonbedrock/mantle` | Converse chat, Titan, Cohere and Nova embeddings, Nova Canvas images, reranking; SigV4 or Bedrock API key. The native Anthropic Messages API and the OpenAI-compatible Mantle endpoint are subpackages. |
| `angular` | Not applicable | none | Frontend framework bindings. |
| `anthropic` | Present | `pkg/provider/anthropic` | Messages (thinking, citations, cache control, container forwarding), Files, Skills, all 23 provider tools, and the experimental Batch (Message Batches API; `WithBatchResultMaxLineBytes`) and Decision (`Decision`, through `internal/decisionlm`). |
| `anthropic-aws` | Present | `pkg/provider/anthropicaws` | Claude Platform on AWS: Anthropic Messages models, Files and Skills; SigV4 (every request, including Skills `GET`s) or API key. |
| `assemblyai` | Present | `pkg/provider/assemblyai` | Transcription (upload, submit, poll). |
| `azure` | Present | `pkg/provider/azure` | Responses (default), Chat, Completion, Azure-hosted DeepSeek, Embedding, Image (including MAI-Image), Speech and Transcription (Azure Speech for MAI-Voice and MAI-Transcribe), and the Azure OpenAI tools. `StreamTranscribe` for MAI-Transcribe-2-Streaming and OpenAI realtime models. |
| `baseten` | Present | `pkg/provider/baseten` | OpenAI-compatible chat and embeddings for Model APIs and dedicated deployments. |
| `black-forest-labs` | Present | `pkg/provider/blackforestlabs` | FLUX Image (generate and edit) and FLUX 3 Video (text-, image- and video-to-video, keyframes, draft enhance; one-call `Generate` and start/status). |
| `bytedance` | Present | `pkg/provider/bytedance` | Seedream Image and Seedance Video (start/status; a webhook URL is forwarded as `callback_url` without `VideoWebhookHandler`, as upstream). |
| `cartesia` | Present | `pkg/provider/cartesia` | Speech, Transcription (unary and `StreamTranscribe`) and the experimental Realtime Ink model (access-token client secrets, binary audio frames, turn or plain transcription WebSocket). |
| `cerebras` | Present | `pkg/provider/cerebras` | OpenAI-compatible chat with Cerebras' request options, reasoning history format and structured output repairs. |
| `code-mode` | Partial | Nested module `codemode/` | `NewTool` (`createCodeModeTool`), `NewCallerTool` (`codeModeTool`), `Run` (`runCodeMode`), `SetMaxWorkers` (`setMaxWorkers`) on `github.com/dop251/goja`: TypeScript programs stripped with `github.com/evanw/esbuild` (syntax errors reported to the model with their location) and TypeScript prompts matching upstream (`codemode/prompts/*.md`), TypeScript declarations of host tools, top-level `await`, concurrent host calls started when first awaited, JSON-only boundary, input validation, output types from `ai.Tool.OutputSchema`, approval and interrupt modes, execution policy limits, console capture and a process-wide worker cap that rejects excess invocations. Continuations interoperate with upstream's: the same signed envelope (`localeCompare`-sorted canonical JSON, HMAC-SHA256) and the token of `run@2.0.0` (`run-replay-v2` ledger, `run-js-v1` values, HS256 codec keyed by the SHA-256 of the signing key, scope hash over upstream's wrapper source), replayed in settlement order with upstream's deterministic `Date` and `Math.random`. Missing: the memory limit (goja cannot bound a heap) and the byte-based stack limit (a call depth instead); the default worker cap is upstream's upper bound of 32, not derived from free memory. |
| `codemod` | Not applicable | none | JavaScript source migration tooling. |
| `cohere` | Present | `pkg/provider/cohere` | Chat (tools, documents, citations, thinking), Embedding, Reranking. |
| `deepgram` | Present | `pkg/provider/deepgram` | Speech and pre-recorded Transcription (upstream has no `doStream`). |
| `deepinfra` | Present | `pkg/provider/deepinfra` | OpenAI-compatible chat, completion and embeddings, and native image inference. |
| `deepseek` | Present | `pkg/provider/deepseek` | Chat (thinking, reasoning effort, image input, assistant prefix completion) and Files (`model.Files`, inline uploads). |
| `devtools` | Present | `pkg/devtools`, `pkg/devtools/cmd/devtools` | `NewTelemetry`, `NewMiddleware`, the upstream JSON serializer and `.devtools/generations.json` store (`OpenStore`), and `NewServer`/`Server.ListenAndServe` (`net/http` port of the viewer API with server-sent events and host and origin checks). Production builds are guarded with the `production` build tag (and `NODE_ENV=production`). The server embeds upstream's React viewer verbatim: the built `dist/client` files of the published `@ai-sdk/devtools@1.0.49` package, whose `src` matches the pinned reference, with its Apache-2.0 `LICENSE`. It serves them like upstream: `/assets/*` with upstream's content types, the viewer page for every other `GET`, and the `AI_SDK_DEVTOOLS_DEV` development page. |
| `elevenlabs` | Present | `pkg/provider/elevenlabs` | Speech and Transcription, with `StreamTranscribe` for `scribe_v2_realtime`. |
| `fal` | Present | `pkg/provider/fal` | Image (generate and edit), Speech, Transcription, Video (queue start/status with `fal_webhook` support). |
| `fireworks` | Present | `pkg/provider/fireworks` | OpenAI-compatible chat, completion and embeddings, and image workflows. |
| `fish-audio` | Present | `pkg/provider/fishaudio` | Speech and Transcription. |
| `gateway` | Present | `pkg/provider/gateway` | Chat, Embedding, Image, Reranking, Speech, Transcription (unary and stream), Video (one-call, start/status, `callbackUrl` webhook), model catalog, credits, spend reports, generation info, and the experimental Batch, Decision and Realtime models. |
| `gladia` | Present | `pkg/provider/gladia` | Pre-recorded transcription (upstream has no `doStream`). |
| `gmicloud` | Present | `pkg/provider/gmicloud` | OpenAI-compatible chat with GMI Cloud's nested error diagnostics. |
| `google` | Present | `pkg/provider/google` | Gemini `generateContent` (`ChatModel`), the Interactions language model and agents, Embedding, Image, Speech, unary and live (`StreamTranscribe`) Transcription, Files, Veo Video, all 7 provider tools (`Search`, `EnterpriseWebSearch`, `Maps`, `URLContext`, `FileSearch`, `CodeExecution`, `VertexRAGStore`), and the experimental Batch, Decision, Realtime (Live API) and Speech translation models. The deprecated `evaluationModel` alias is not ported. |
| `google-vertex` | Present | `pkg/provider/googlevertex`, `googlevertex/anthropic`, `googlevertex/maas`, `googlevertex/xai` | Gemini models (`google.ChatModel`), Interactions, Embedding, Image, Speech (Gemini and Chirp 3: HD), Transcription (Gemini and Cloud Speech-to-Text, live `StreamTranscribe`), Veo Video; OAuth token source or express-mode API key. Anthropic, MaaS and xAI partner models are subpackages. |
| `groq` | Present | `pkg/provider/groq` | Chat (reasoning, structured outputs, service tiers, `BrowserSearch` tool) and Whisper Transcription. |
| `harness` | Present | `pkg/harness`, `pkg/harness/bridge`, `pkg/harness/credential` | Harness specification v1 with optional capability interfaces, `NetworkSession`, `Agent` (`HarnessAgent`: sessions, bootstrap recipes, `Generate`/`Stream`/`Continue*`/`Steer`, `PrepareCall`, typed output, `StopWhen` suspension, permission modes, tool approval, host tools with the restricted sandbox in the tool's context and preliminary results through `ai.ExecuteTool`), `AgentSession`, `StreamResult` (including `ElementStream`), sandbox templates, diagnostics and reporters, and the host side of the bridge protocol. The in-sandbox bridge runtime stays JavaScript: each adapter embeds the published bundle under `pkg/harness/<adapter>/assets`. Adaptations: `Agent` does not implement `ai.Agent` (turns need a session); `callOptionsSchema` is replaced by type assertions in `PrepareCall`; the deprecated sandbox provider API is not ported. Preliminary host tool results are streamed but, as in `pkg/ai`, not recorded in step content. |
| `harness-acp` | Present | `pkg/harness/acp` | `New` (`createACP`) with every upstream setting, profile identities, lifecycle schema, skills, approvals, structured output and recovery; bundle from `@ai-sdk/harness-acp` 1.0.96. Built-in tool filtering and manual compaction report `CapabilityUnsupportedError`, as upstream. |
| `harness-claude-code` | Present | `pkg/harness/claudecode` | `New` (`createClaudeCode`) with every upstream setting, native subscription login, 48 built-in tools, approvals and filtering, skills, `/compact`, resume and recovery; bundle from `@ai-sdk/harness-claude-code` 1.0.161. |
| `harness-cline` | Blocked | none | No sandbox protocol carries host tools. Upstream runs `@cline/agents` in the host Node.js process. The Cline CLI (`cline` 3.0.73, the latest release) offers two machine-readable modes. `--json` runs one non-interactive task, with tool approval fixed by `--auto-approve`. `--acp` speaks ACP v1 with approvals, cancellation and `session/load`, but its `AcpAgent` stores the `mcpServers` of `session/new` and `session/load` without passing them to the Cline core, and its `prompt` keeps only text blocks (`apps/cli/src/acp/acpAgent.ts` at tag `cli-v3.0.73`, unchanged on `main`). The ACP bridge creates the host tool MCP server at run time with a per-session relay URL and credential, so it cannot be preconfigured in Cline's own settings. |
| `harness-codex` | Present | `pkg/harness/codex` | `New` (`createCodex`) with every upstream setting, ChatGPT subscription login (file and keyrings), built-in tools and filtering, skills, thread resume and recovery; bundle from `@ai-sdk/harness-codex` 1.0.159. Manual compaction and non-`allow-all` permission modes report `CapabilityUnsupportedError`, as upstream. |
| `harness-cursor` | Present | `pkg/harness/cursor` | `New` (`createCursor`) over `pkg/harness/acp`: 30 built-in tools, MCP tool detection, credential brokering and the native Cursor CLI login. |
| `harness-deepagents` | Present | `pkg/harness/deepagents` | `New` (`createDeepAgents`) with every upstream setting, 9 built-in tools, approvals, skills, structured output and recovery; bundle from `@ai-sdk/harness-deepagents` 1.0.157. A respawned bridge replays the suspended turn from its event log, where upstream starts fresh. |
| `harness-fx` | Partial | `pkg/harness/fx` | `New` (`createFx`) over `pkg/harness/acp`, driving `fx acp` from the pinned fx v0.0.14 release (the engine of upstream's `libfx` 0.0.14), installed with checksum verification. Includes AI Gateway authentication with credential brokering and headers, `Effort`, `ServiceTier`, MCP servers, 17 native built-in tools (`glob`, `grep` and `webSearch` under common names), host tools, approvals by permission mode (fx runs in its `ask` mode), images, instructions (`~/.fx/AGENTS.md`), skills (`~/.fx/skills`) and resume. Upstream runs `libfx` on the host with its own sandbox tools; here fx's native tools run in the sandbox. Missing: the subscription fallback (fx's ACP server reads fx, Codex and Grok logins only from its own files and would rotate their refresh tokens in the sandbox), structured output and steering (no ACP mapping in the bridge), `askUserQuestions` (fx answers questions only in its interactive shell), built-in tool filtering (ACP v1), the `backend` setting (no `libfx`), and AI Gateway base URL overrides (fx accepts only loopback overrides). |
| `harness-github-copilot` | Present | `pkg/harness/githubcopilot` | `New` (`createGitHubCopilot`) over `pkg/harness/acp`: pinned Copilot CLI project from `@ai-sdk/harness-github-copilot` 1.0.53, 16 built-in tools, credential brokering and native logins. |
| `harness-grok-build` | Present | `pkg/harness/grokbuild` | `New` (`createGrokBuild`) over `pkg/harness/acp`: pinned Grok Build CLI from `@ai-sdk/harness-grok-build` 1.0.95, 24 built-in tools plus `askUserQuestions`, structured output and the native Grok CLI login. |
| `harness-opencode` | Present | `pkg/harness/opencode` | `New` (`createOpenCode`) with every upstream setting and provider, native logins, 12 built-in tools, approvals and filtering, skills, manual compaction, resume and recovery; bundle from `@ai-sdk/harness-opencode` 1.0.159. |
| `harness-pi` | Partial | `pkg/harness/pi` | `New` (`createPi`). Upstream runs the Pi SDK in the host process; Go runs Pi 1.0.3's RPC mode (`pi --mode rpc`) inside the sandbox, installed by a locked pnpm bootstrap. Sandbox processes have no stdin, so an in-sandbox relay feeds numbered inbox files to Pi and appends its strict-JSONL stdout to an outbox the host follows by byte offset. A Pi extension replaces the built-ins with upstream's seven tool schemas (native implementations, filtering, `fileToolPathPolicy`, file-change events), registers each turn's host tools and asks the host for their results and for built-in approvals through RPC extension UI input dialogs, and applies instructions, providers and MCP servers (direct/deferred, `tool_search`). Covers auth modes with brokering, AI Gateway default model and resolver, Pi subscription `auth.json` or a `CredentialStore` (rotated tokens saved back), thinking level, steering, abort, manual and auto compaction, usage, session files (`--session-dir`, `--session`), lifecycle schema and stderr/extension diagnostics. Detach and SuspendTurn keep Pi running and any later process reattaches losslessly (`ReattachLiveRuntime`, replacing in-process parking). Gaps: `cacheRetention: none` (no RPC option), cold continuation of a tool batch waiting on host input when Pi's process is gone (upstream's journal replay needs in-process agent access; Go re-drives the thread), inline extension factories (Go loads extension source files), deprecated `mcpSettings`, and denied roots nested under a grep/glob search directory are not filtered. |
| `heygen` | Present | `pkg/provider/heygen` | Text-, image- and reference-to-video (start/status). |
| `huggingface` | Present | `pkg/provider/huggingface` | Responses language model through the Hugging Face router. |
| `hume` | Present | `pkg/provider/hume` | Expressive Speech. |
| `klingai` | Present | `pkg/provider/klingai` | Text-, image-, reference-to-video and motion control (start/status); API key or access/secret key JWT. |
| `langchain` | Not applicable | none | Adapter for a JavaScript library. |
| `llamaindex` | Not applicable | none | Adapter for a JavaScript library. |
| `luma` | Present | `pkg/provider/luma` | Photon Image (queued and polled; URL image inputs only, as upstream). |
| `mcp` | Partial | `pkg/mcp` | `NewClient` (`createMCPClient`) with discovery and `initialize` fallback, capabilities, cancellation, tools (`Tools`, `ToolsFromDefinitions`, typed schemas, output validation, `x-mcp-header`), resources, prompts, completions, elicitation, ping, notifications, MCP Apps, the Streamable HTTP, SSE and stdio transports, and OAuth (`Auth`, discovery, dynamic registration, PKCE, refresh, with token and registration requests on a DNS-validating client). Experimental events (`ClientOptions.ExperimentalEvents`): `Client.Events` (direct webhook subscriptions persisted in an `EventStore`, with generated Standard Webhooks secrets and routing keys), `NewEventWebhook` (`createMCPEventWebhook`: verification challenges, HMAC signature checks, gaps and terminations) and `Client.ManagedEvents` over an `EventAdapter`. The stdio `Shell` and `ShellPath` options run the command through `/bin/sh -c` or `cmd.exe /d /s /c` with every argument quoted, where upstream joins arguments unquoted. Missing: the mock transport (test infrastructure). |
| `minimax` | Present | `pkg/provider/minimax` | Messages chat (built on `anthropic`) and H3 Video (one-call and start/status). |
| `mistral` | Present | `pkg/provider/mistral` | Chat, Conversations with `WebSearch()` and `WebSearchPremium()`, Embedding, Speech, Transcription. |
| `moonshotai` | Present | `pkg/provider/moonshotai` | Kimi and moonshot-v1 chat with thinking controls, partial mode and dynamic tool loading. |
| `open-responses` | Present | `pkg/provider/openresponses` | Responses language model for any Open Responses endpoint (`New(name, url)`), custom tools. |
| `openai` | Present | `pkg/provider/openai` | Responses (default), Chat, Completion, Embedding, Image, Speech, Transcription (realtime models stream), Files, Skills, all 13 provider tools, and the experimental Batch, Decision, Realtime (`Realtime`, `Live`) and Speech translation models. |
| `openai-compatible` | Present | `pkg/provider/openaicompatible` | Chat, Completion, Embedding, Image, with exported hooks and model constructors for providers built on it. |
| `otel` | Present | Nested module `otel/` | `otel.New` (GenAI semantic conventions: agent, step, chat, tool, embedding, rerank, speech, transcription, translation and decision spans with `gen_ai.*` attributes and SemConv messages) and `otel.NewLegacy` (the earlier `ai.*` layout, including `ai.decide` and `ai.decide.doDecide` spans). Both implement `ai.DecisionTelemetry`; `Options.ExperimentalDecision` records decision state, questions and answers. Spans are children of the span in the call's context; a nil tracer uses the global tracer provider. Not applicable: object generation spans and the `schema` option (deprecated operations), and the `runtimeContext` option (`EnrichSpan` receives the `context.Context`). |
| `perplexity` | Present | `pkg/provider/perplexity` | Agent language model, Embedding, and the experimental Decision model. |
| `policy-opa` | Present | `pkg/policyopa`, nested module `policyopawasm/` | `HTTPPolicyClient`, `OPAPolicy`, `OptionalOPAPolicy`, `Shadow` and `WrapMCPTools` (both accept an `ai.ToolApprovalFunc` or a per-tool `ai.ToolApprovals` map) and `OPACapabilityMiddleware`, failing closed except on canceled contexts. `policyopawasm.New` (`wasmPolicyClient`) runs `opa build -t wasm` bundles on `github.com/tetratelabs/wazero` with the `@open-policy-agent/opa-wasm` ABI: ABI 1.x version checks, the `opa_eval` fast path of ABI 1.2 and the evaluation-context path of earlier versions, data documents, named entrypoints, memory limits, `opa_abort`/`opa_println`, and the `json.is_valid`, `regex.split`, `sprintf`, `yaml.is_valid`, `yaml.marshal` and `yaml.unmarshal` built-ins (Go semantics, as in OPA; YAML through `go.yaml.in/yaml/v3`) plus custom built-ins. Evaluations are canceled with their context and use pooled instances. |
| `prodia` | Present | `pkg/provider/prodia` | Image, the multimodal language model, and Video. |
| `quiverai` | Present | `pkg/provider/quiverai` | Arrow SVG image models and Arrow language models (built on `openresponses`). |
| `react` | Not applicable | none | Frontend framework bindings. Go servers interoperate with them through `pkg/ai/ui`. |
| `replicate` | Present | `pkg/provider/replicate` | Image (synchronous wait and polling) and Video (start/status with webhook support). |
| `revai` | Present | `pkg/provider/revai` | Asynchronous transcription. |
| `rsc` | Not applicable | none | React Server Components bindings. |
| `sandbox-just-bash` | Partial | `sandboxshell/` (nested module) | `sandboxshell.New` creates a fresh in-memory sandbox per call and applies the optional harness template; bash runs in-process with the `mvdan.cc/sh` interpreter over a concurrency-safe virtual filesystem, with Go implementations of just-bash's common commands (file, text, `grep`, `sed`, `find`, `xargs`, `bash`/`sh` and others). Host files, host programs and the network are never reached; ports, network policy and `Resume` fail with `harness.CapabilityUnsupportedError`, as upstream. Not ported: the native just-bash adapters and deprecated provider, host-backed overlay filesystems, custom commands, execution limits, links, and the optional `curl`, Python, JavaScript, SQLite and `jq` commands. `pkg/sandbox/local` remains a real, non-isolated shell. |
| `sandbox-vercel` | Partial | `pkg/sandbox/vercel` | `Create`, `Resume` and a `harness.NetworkSession` over the Vercel Sandbox REST API (files, commands with log streams, detached processes, ports, network policy with request transformations), with OIDC or token authentication. OIDC tokens come from `VERCEL_OIDC_TOKEN_FILE` or `VERCEL_OIDC_TOKEN` and are refreshed like `@vercel/oidc` when missing or expiring within five minutes of a request: the cached project token, or a new one issued with the Vercel CLI's `auth.json` access token (refreshed through OAuth) or, for keyring storage, `vercel project token`. Requests retry network errors, 429 and 5xx twice with randomized backoff from 400 ms, honoring `Retry-After` up to 20 seconds, as `@vercel/sandbox` does. Adaptations: a failed CLI token refresh leaves `auth.json` unchanged, and the CLI is looked up in `PATH` only. Development credentials port `@vercel/sandbox` 3.1.0: `SignIn` runs the Vercel CLI's OAuth device sign-in, showing the code through a caller-supplied `Prompt` and polling with the context, and stores the tokens in the CLI's `auth.json`; `DevelopmentCredentials` uses or refreshes that token and infers the team and project (linked project, or the created `vercel-sandbox-default-project` in the first usable team), cached per scope; `ConnectionOptions.DevelopmentSignIn` enables the fallback when no OIDC token is available. Adaptations: the caller opts in instead of a TTY check, nothing opens a browser, and `auth.json` keeps its other members and is replaced atomically. Missing: the native `Sandbox` adapters (no Go SDK). Not applicable: the deprecated `createVercelSandbox` provider. |
| `svelte` | Not applicable | none | Frontend framework bindings. |
| `test-server` | Deferred | none | Test infrastructure; deferred with the test decision. |
| `togetherai` | Present | `pkg/provider/togetherai` | OpenAI-compatible chat, completion and embeddings, Image and Reranking. |
| `tool-bash` | Present | `pkg/tools/bash` | `New` (`createBashTool`) over any `sandbox.Session`. |
| `topaz` | Present | `pkg/provider/topaz` | Image and Video enhancement. |
| `tui` | Present | Nested module `tui/` | `tui.RunAgent` (`runAgentTUI`): streaming text and reasoning, tool cards, `y`/`n` approvals, markdown, scrollback, statistics and display modes, for a local `Agent` or a remote agent through `Transport` (a `ui.ChatTransport`; approvals continue the last assistant message, and statistics come from its metadata). Adaptation: the sandbox option is not ported, because package `ai` has no sandbox call option. |
| `typesafe-ai` | Present | `pkg/provider/typesafeai` | Experimental System One Decision model. |
| `valibot` | Not applicable | none | Schema library adapter; Go types replace schema libraries. |
| `voyage` | Present | `pkg/provider/voyage` | Embedding and Reranking. |
| `vue` | Not applicable | none | Frontend framework bindings. |
| `workflow` | Blocked | none | Built on the Vercel Workflow DevKit (`'use step'` directives, `getWritable`, `createHook`, `start`). The DevKit has JavaScript, TypeScript and Python (beta) SDKs and no Go SDK. |
| `workflow-harness` | Blocked | none | Runs `HarnessAgent` as a Workflow DevKit workflow; blocked on a Go Workflow SDK, as `workflow` is. |
| `xai` | Present | `pkg/provider/xai` | Responses with the server-side tools, Image (generate and edit), Speech, Transcription (file and `StreamTranscribe`), Files, Video (generation, edit, extension, reference-to-video), and the experimental Batch and Realtime models. |
| `zai` | Present | `pkg/provider/zai` | OpenAI-compatible GLM chat. |

Go-only provider: `pkg/provider/openrouter` (OpenRouter, built on `openaicompatible`, with reasoning details, usage accounting, routing and plugins). Upstream's equivalent is the community package `@openrouter/ai-sdk-provider`.

Streaming transcription (`model.StreamingTranscriptionModel`) is implemented wherever the upstream transcription model has `doStream`: `openai` (realtime models), `azure`, `elevenlabs`, `cartesia`, `google`, `googlevertex`, `gateway` and `xai`.

## Core modules: `packages/ai/src`

| Upstream | Status | Go location | Notes |
| --- | --- | --- | --- |
| `generate-text` (`generateText`) | Present | `pkg/ai/generate_text*.go`, `text_generation.go`, `step_result.go`, `tool_*.go`, `prepare_step.go`, `stop_condition.go`, `response_messages.go`, `language_model_events.go`, `timeout.go` | Multi-step tool loop, `StopWhen`, `PrepareStep`, active tools and `FilterActiveTools`, tool order and concurrency, repair, input refinement, approvals (`ToolApproval` as an `ai.ToolApprovalFunc` or a per-tool `ai.ToolApprovals` map, HMAC-signed), tool fingerprints, `PruneMessages`, include flags, tool callers, tool search, language model call callbacks, telemetry, `LogWarnings` and `Timeout`. The sandbox option is a context value (see adaptations). |
| `generate-text` (`streamText`) | Present | `pkg/ai/stream_text*.go`, `stream_part.go`, `stream_retry.go`, `stream_language_model_call.go`, `smooth_stream.go`, `text_events.go` | Full stream parts, `SmoothStream` and custom transforms, `OnChunk`, `OnError`, `OnAbort`, partial output and element streams, `StreamRetries`, timeouts, `StreamTextLowMemory` and `StreamLanguageModelCall`. Single consumer, no tee (see adaptations). |
| `generate-text/output` | Present | `pkg/ai/output.go` | `TextOutput`, `ObjectOutput[T]`, `ArrayOutput[T]`, `ChoiceOutput`, `JSONOutput`, and a custom `Output` interface. |
| `generate-object` | Not applicable | none | Deprecated upstream; `Output` on `GenerateText` and `StreamText` replaces it. |
| `agent` | Present | `pkg/ai/agent.go`, `tool_loop_agent.go`, `pkg/ai/ui/agent_stream.go` | `Agent` interface, `ToolLoopAgent` (20-step default with a warning), `PrepareCall`, per-call `Timeout`, agent UI stream helpers. `infer-*` are TypeScript-only types. |
| `embed` | Present | `pkg/ai/embed.go`, `embed_events.go` | `Embed`, `EmbedMany` (batching, parallel calls, dimension validation), telemetry and `LogWarnings`. |
| `generate-image` | Present | `pkg/ai/image.go` | Generation, editing (`Images`, `Mask`), per-call splitting, `LogWarnings`. |
| `generate-speech` | Present | `pkg/ai/speech.go`, `speech_events.go` | Telemetry and `LogWarnings`. |
| `transcribe` | Present | `pkg/ai/transcribe.go`, `stream_transcribe.go`, `transcription_events.go` | `Transcribe` with URL download, and experimental `StreamTranscribe` for `model.StreamingTranscriptionModel`. Telemetry and `LogWarnings`. |
| `rerank` | Present | `pkg/ai/rerank.go`, `rerank_events.go` | Generic over the document type; telemetry and `LogWarnings`. |
| `generate-video` | Present (experimental) | `pkg/ai/generate_video.go`, `video_operation.go`, `start_video.go` | `GenerateVideo`, `StartVideo`, `GetVideoStatus`: per-call splitting, URL download, polling (`VideoPollOptions`) or webhooks (`model.VideoWebhookFactory`). |
| `translate` | Present (experimental) | `pkg/ai/stream_translate.go` | `StreamTranslate` for `model.SpeechTranslationModel`. |
| `realtime` | Present (experimental, server side) | `pkg/ai/realtime_*.go` | `ConnectRealtime` over `internal/websocket` (relay, server WebSocket and client-secret connections, reducer, command tracker, tool calls with automatic follow-up, text or binary frames per `model.RealtimeFrame`), `RealtimeToolDefinitions`, `CreateRealtimeSetup`, audio encode, decode and resample helpers. Not applicable: browser audio capture and playback, the WebRTC transport and the browser send-buffer budget; browsers use upstream's client against a Go setup endpoint. |
| `decide` | Present (experimental) | `pkg/ai/decide*.go`, `decision_language_model.go` | `Decide` with validation, refusals, retries, callbacks, `LogWarnings` and the optional `DecisionTelemetry`; `NewDecisionLanguageModel`. The deprecated `experimental_evaluate` aliases are not ported. |
| `batch` | Present (experimental) | `pkg/ai/batch*.go` | `StartBatch`, `GetBatchStatus`, `GetBatchResults`, `CancelBatch`, `ListBatches` with persisted `BatchReference`. |
| `upload-file` | Present | `pkg/ai/upload_file.go` | `UploadFile` over a `model.Files` (`provider.Files()` or `Registry.Files`). |
| `upload-skill` | Present | `pkg/ai/upload_skill.go` | `UploadSkill` over a `model.Skills`. |
| `tool-search` | Present | `pkg/ai/tool_search.go`, `tool_caller.go` | `ai.ToolSearch` (keyword ranking or custom function) and `Tool.DeferLoading`. |
| `middleware` | Present | `pkg/ai/middleware` | `WrapLanguageModel`, `WrapEmbeddingModel`, `WrapImageModel`, `WrapProvider`, `DefaultSettings`, `DefaultInstructions`, `DefaultEmbeddingSettings`, `ExtractJSON`, `ExtractReasoning`, `SimulateStreaming`, `AddToolInputExamples`. |
| `registry` | Present | `pkg/ai/registry.go` | `NewRegistry`, `NewCustomProvider`, `NoSuchProviderError`, and lookups for every model kind plus `Batch`, `Files` and `Skills`. |
| `prompt` | Present | `pkg/ai/prompt*.go`, `download.go` | Standardization, validation, system-message guard, URL download, media type detection. `wrap-gateway-error` is not needed (no global gateway). |
| `model` (`resolve-model`, `as-*-v3/v4`) | Not applicable | none | One contract version and no global provider for string model IDs. |
| `types` | Present | `pkg/model`, `pkg/ai` | Results, usage, warnings and metadata types. |
| `error` | Present | `pkg/ai/errors.go`, `retry.go`, `pkg/model/errors.go`, `pkg/model/decision.go` | Generation, tool, approval, stream, retry and no-output errors, including video, translation and decision errors. `UnsupportedModelVersionError` does not apply. |
| `text-stream` | Present | `pkg/ai/ui/text_stream.go` | `CreateTextStreamResponse`, `PipeTextStreamToResponse`. |
| `ui-message-stream` | Present | `pkg/ai/ui` | Chunks in the upstream JSON wire format, `ToMessageStream`, `CreateMessageStream` and `Writer`, SSE responses, `ReadMessageStream`. `ValidateMessages` checks available tool outputs against `ai.Tool.OutputSchema`. |
| `ui` | Present (server side) | `pkg/ai/ui` | `Message` and parts, `ValidateMessages`, `ConvertToModelMessages`, last-assistant-message helpers, and the HTTP chat transports for Go clients (`ChatTransport`, `NewDefaultChatTransport`, `NewTextStreamChatTransport`: send, reconnect with 204 for no active stream, preparation hooks, cancellation through the context). Not applicable: client `Chat`, the direct and WebSocket chat transports, and `useCompletion` (browser code). |
| `telemetry` | Present | `pkg/ai/telemetry.go`, `telemetry_operations.go` | `ai.Telemetry` (`OnError`) with optional per-operation interfaces (`TextTelemetry`, `EmbedTelemetry`, `RerankTelemetry`, `SpeechTelemetry`, `TranscriptionTelemetry`, `TranslationTelemetry`, `DecisionTelemetry`; embed `ai.NopTelemetry` for all), `TelemetryOptions` and the `ExecuteLanguageModelCall`/`ExecuteTool` wrappers. Not applicable: the global registry and the Node diagnostics channel. |
| `logger` | Present | `pkg/ai/log_warnings.go` | `LogWarningsFunc`, `WarningsEvent`, `ai.SlogWarnings`, per call. Not applicable: the global `AI_SDK_LOG_WARNINGS` switch. |
| `util` | Present | `internal/*`, `pkg/ai/retry.go`, `download.go`, `cosine_similarity.go` | Retry with backoff, partial JSON repair, merging, IDs, data URLs, downloads, `CosineSimilarity`. Stream and promise helpers are not needed in Go. |
| `global.ts`, `version.ts` | Not applicable | none | No global provider and no user agent. |
| `test` (mock models) | Deferred | none | Test utilities; waits for the owner's decision on tests. |

## Provider contracts: `packages/provider/src`

| Model kind | Status | Go location |
| --- | --- | --- |
| `language-model` | Present | `model.LanguageModel` (`Generate`, `Stream`, `SupportedURLs`), `CallOptions`, prompt, content, stream parts, usage, finish reason |
| `embedding-model` | Present | `model.EmbeddingModel` |
| `image-model` | Present | `model.ImageModel` |
| `speech-model` | Present | `model.SpeechModel` |
| `transcription-model` | Present | `model.TranscriptionModel`, plus the optional `model.StreamingTranscriptionModel` |
| `reranking-model` | Present | `model.RerankingModel` |
| `video-model` | Present (experimental) | `model.VideoModel` with the optional `VideoGenerator`, `VideoOperator` and `VideoWebhookHandler`; `model.VideoProvider` |
| `realtime-model` | Present (experimental) | `model.RealtimeModel` (`ParseServerEvent`, `SerializeClientEvent` returning a text or binary `*model.RealtimeFrame`, `BuildSessionConfig`, `Capabilities`) with optional connection, client-secret and WebRTC interfaces; `model.RealtimeProvider` |
| `speech-translation-model` | Present (experimental) | `model.SpeechTranslationModel`; `model.SpeechTranslationProvider` |
| `decision-model` | Present (experimental) | `model.DecisionModel`; `model.DecisionProvider` |
| `batch` | Present (experimental) | `model.Batch` with the optional `BatchCanceller` and `BatchLister`; `model.BatchProvider` |
| `files` | Present | `model.Files` with the optional `FileMetadataReader`, `FileDownloader` and `FileDeleter`; `model.FilesProvider` |
| `skills` | Present | `model.Skills`; `model.SkillsProvider` |
| `provider` | Present | `model.Provider` (six model kinds; a missing kind returns an error wrapping `ErrNoSuchModel`). `model.Kind` names model kinds (`model.LanguageModelKind` and the others) in errors and registries. |
| `language-model-middleware`, `embedding-model-middleware`, `image-model-middleware` | Present | `pkg/ai/middleware` (`Language`, `Embedding`, `Image`) |
| `shared` | Present | `model.Warning`, `ProviderOptions`, `ProviderMetadata`, `http.Header` |
| `json-value` | Present | `any`, `json.RawMessage` |
| `errors` | Present | `model.APICallError` plus sentinel errors |
| v2 and v3 contract versions | Not applicable | Go has one contract version |

## Provider utilities: `packages/provider-utils/src`

| Upstream piece | Status | Go location |
| --- | --- | --- |
| `post-to-api`, `get-from-api`, `delete-from-api`, response handlers, header helpers, `handle-fetch-error` | Present | `internal/apicall` |
| `convert-to-form-data`, multipart and stream uploads | Present | `internal/apicall` |
| `parse-json-event-stream`, SSE parsing | Present | `internal/sse`, `internal/chatsse` |
| `retry-with-exponential-backoff`, `delay` | Present | `pkg/ai/retry.go` |
| `schema`, `to-json-schema`, `validate-types` | Present (Go types) | `internal/schema` |
| `secure-json-parse`, `parse-json` | Present | `encoding/json` with typed decoding, `internal/jsonobject` |
| `generate-id` | Present | `internal/id` |
| `detect-media-type`, `resolve-full-media-type`, `media-type-to-extension` | Present | `internal/mediatype` |
| `download-blob`, `fetch-untrusted-url`, `validate-download-url`, `read-response-with-size-limit` | Present | `internal/download` (SSRF guards, size limits) |
| `streaming-tool-call-tracker` | Present | `internal/toolcalls` |
| `map-reasoning-to-provider` | Present | `internal/reasoning` |
| `create-tool-name-mapping` | Present | `internal/toolnames` |
| `tool` types (`tool`, `dynamicTool`, `outputSchema`), `execute-tool` | Present | `ai.Tool` (`OutputSchema`), `ai.NewTool` (schemas generated from the input and output types), `ai.DynamicTool`, `ai.ExecuteTool` (preliminary outputs through `OnPreliminary`) |
| `provider-defined-tool-factory`, `provider-executed-tool-factory` | Present | `model.ProviderTool`, `ai.ProviderDefinedTool`, `ai.ProviderExecutedTool` |
| `inject-json-instruction` | Present | Ported inside the providers that use it (for example `mistral`, `amazonbedrock`) and the `ExtractJSON` middleware |
| `load-api-key`, `load-setting` | Present | Per-provider environment fallback, `model.ErrLoadAPIKey` |
| `is-provider-reference`, `resolve-provider-reference` | Present | `model.FileReference` |
| `websocket`, `connect-to-websocket` | Present | `internal/websocket` (RFC 6455 client, no compression extension), `internal/livetranscription` (transcription and speech translation sessions), `internal/geminilive` |
| `transcription-stream-envelope` | Present | `internal/transcriptionstream` |
| `decision-language-model` | Present (experimental) | `internal/decisionlm`, `ai.NewDecisionLanguageModel` |
| sandbox session types (`Experimental_SandboxSession`) | Present | `pkg/sandbox` |
| `convert-inline-file-data-to-uint8-array`, `normalize-batch-request-counts` | Present | `model.InlineFileBytes`, `model.NormalizeBatchRequestCounts` |
| `with-user-agent-suffix`, `get-runtime-environment-user-agent`, `is-browser-runtime`, `serialize-model-options` | Not applicable | No user agent, no browser runtime, and options are Go values |
