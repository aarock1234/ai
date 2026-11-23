# Roadmap

This is the state of the port against `ai@7.0.146` and `@ai-sdk/provider@4.0.31`, and the work that remains. The rewrite shipped as `v0.4.0` and the current release is `v0.6.1`; see the [changelog](../CHANGELOG.md). Statuses and Go locations are in [feature parity](feature-parity.md). Work follows the pinned upstream source, not older plans.

## Current state

- Phases A to E are complete. Every upstream package is ported, adapted, or recorded as not applicable to Go, with the gaps listed below.
- The core module and the nested modules `otel/`, `codemode/`, `tui/`, `policyopawasm/` and `sandboxshell/` pass the static checks listed in the [code audit](code-audit.md).
- The programs in `examples/` and the README use the current API, and the examples build. `examples/openrouter` and `examples/decide` have been run; no tests cover the rewritten code.

## Live runs

`examples/openrouter` ran on 2026-11-17 against `anthropic/claude-opus-5.5` and `openai/gpt-6.1-terra` through OpenRouter, with data collection denied and usage accounting on. Every check passed on both models:

- `GenerateText` with a system prompt.
- `ObjectOutput` with nested objects and arrays, enums, a nullable pointer, an `omitempty` field, and `minLength`, `maxLength`, `minimum`, `minItems` and `maxItems`, all under strict structured output.
- `ArrayOutput` with bounds, enums and nullable fields.
- A `ToolLoopAgent` with two tools and `IsStepCount(5)`, which called both tools over two steps.
- The cost of every step, read from `openrouter.ChatMetadata`.
- A model ID that does not exist, which failed with a `*model.APICallError` (status 400, not retryable).

`examples/openrouter` also passed on `openai/gpt-6-luna`, and `examples/decide` ran `Decide` against OpenAI's Decisions API (`POST /v1/decisions`) with `gpt-6-luna`: a choice question and a boolean question per message, answered with full probability distributions and input-token usage. It also ran through OpenRouter's own Decisions API (`POST /api/alpha/decisions`) with `scaledown/classify-1`: the same labels, with full distributions.

`examples/openrouter`'s prompt cache check held a two-turn conversation in one `SessionID` with automatic caching (`CacheControl`): the second turn read 4,475 of 4,592 input tokens from the cache on `anthropic/claude-opus-5.5`, and 2,791 of 2,831 on `openai/gpt-6.1-terra`.

The OpenRouter run led to two schema fixes in `internal/schema`. Optional fields of model-written values are now required and nullable, so the schemas are valid under the strict structured output that OpenAI-compatible providers use by default; tool outputs keep optional fields optional. The `jsonschema` tag's description may contain commas, and the tag supports `minLength`, `maxLength`, `minimum`, `maximum`, `minItems` and `maxItems`.

## Completed phases

- **A: core.** `generateText` and `streamText` with tool loops, approvals, tool search and callers, stream retries and timeouts, plus telemetry and warning logging for every model kind, and streaming transcription.
- **B: policy and observability.** OpenTelemetry (`otel/`), OPA tool policies (`pkg/policyopa`, with the WebAssembly client in `policyopawasm/`), DevTools (`pkg/devtools`) and the terminal UI (`tui/`).
- **C: media and uploads.** Files, skills and video contracts, `UploadFile`, `UploadSkill`, `GenerateVideo`, `StartVideo` and `GetVideoStatus`, and every provider's files and video models.
- **D: MCP and experimental features.** The MCP client (`pkg/mcp`), batch, decision, realtime (server side) and speech translation, Google Interactions, code mode (`codemode/`), the harness (`pkg/harness`) with all bridge adapters, sandboxes (`pkg/sandbox/vercel`, `pkg/sandbox/local`, `sandboxshell/`) and the bash tool (`pkg/tools/bash`).
- **E: providers.** All 47 upstream provider packages, plus OpenRouter.

## Remaining gaps

These come from the Partial and Deferred rows of [feature parity](feature-parity.md). None blocks a release.

- **Code mode:** a memory limit and a byte-based stack limit, which goja cannot enforce.
- **Vercel Sandbox:** the native `Sandbox` adapters (no Go SDK).
- **Harness adapters:** Pi (`pkg/harness/pi`, over `pi --mode rpc`) cannot honor `CacheRetention` `none`, and a cold continuation (Pi's sandbox process gone) cannot deliver results to a tool batch that was waiting on host input; it re-drives the thread instead. Cline is blocked: its `--json` mode is a one-shot task, and its ACP mode ignores client MCP servers, so host tools cannot reach it. fx (`pkg/harness/fx`, over ACP) lacks the subscription fallback, structured output, steering, `askUserQuestions` and built-in tool filtering.
- **In-memory shell sandbox (`sandboxshell/`):** custom commands, execution limits, links and just-bash's optional `curl`, Python, JavaScript, SQLite and `jq` commands.
- **Workflow:** `workflow` and `workflow-harness` are blocked until the Vercel Workflow DevKit has a Go SDK (it has JavaScript, TypeScript and Python).
- **Test utilities:** upstream's mock models and the MCP mock transport, which depend on the test decision below.

## Phase F: release

- **License.** Done. The SDK is licensed under the Apache License 2.0, matching upstream (`LICENSE`, also copied into each nested module). `NOTICE` attributes the upstream and lists the embedded upstream files under `pkg/harness/*/assets` and `pkg/devtools/client`, which keep their Apache-2.0 `LICENSE` files.
- **Tests.** The pre-rewrite `*_test.go` files were deleted at the owner's request, so the repository has no tests. `make check-static` is the verification until tests are added; `make test` and the CI test step pass with no test files.
- **Nested modules.** `otel/`, `codemode/`, `tui/`, `policyopawasm/` and `sandboxshell/` require the tagged core `v0.4.0`, without a `replace`, and are tagged with their path prefix at the same version (`otel/v0.4.0`, and so on). Local development still builds them against the working core through an untracked `go.work`. A release that changes the core and a nested module together tags the core first, then bumps the nested module's requirement and tags it.
- **Versioning.** The API breaks compatibility with v0.3.0, so the rewrite is tagged v0.4.0 with the [migration guide](migration.md). The `Makefile` holds the only tool pin.
- **CI.** The `Makefile` builds, lints and format-checks all six modules (`MODULES := . otel codemode tui policyopawasm sandboxshell`). CI runs `make workspace`, `make check-static` and `make test`. `make workspace` generates an untracked `go.work`, so the nested modules build against the working core instead of the published version they require. The release workflow reuses the CI job before it publishes the GitHub release. `make test` and the CI test step cover only the root module.
