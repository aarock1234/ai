# Roadmap

This is the state of the port against `ai@7.0.146` and `@ai-sdk/provider@4.0.31`, and the work left before a release. Statuses and Go locations are in [feature parity](feature-parity.md). Work follows the pinned upstream source, not older plans.

## Current state

- Phases A to E are complete. Every upstream package is ported, adapted, or recorded as not applicable to Go, with the gaps listed below.
- The core module and the nested modules `otel/`, `codemode/`, `tui/`, `policyopawasm/` and `sandboxshell/` pass the static checks listed in the [code audit](code-audit.md).
- The programs in `examples/` and the README use the current API, and the examples build. They have not been run.
- Nothing has been run against live provider APIs, and no tests cover the rewritten code.

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
- **Nested modules.** `otel/`, `codemode/`, `tui/`, `policyopawasm/` and `sandboxshell/` require `github.com/aarock1234/ai v0.0.0` through a `replace ../` directive. Before release, each must require a tagged core version and drop the `replace`, and the modules must be tagged with their path prefix (for example `otel/v0.x.y`).
- **Versioning.** The API breaks compatibility with v0.3.0. Tag a new minor version (v0.x) with the [migration guide](migration.md), and keep the CI and release tool pins aligned.
- **CI.** The `Makefile` builds, lints and format-checks all six modules (`MODULES := . otel codemode tui policyopawasm sandboxshell`). CI and the release workflow lint each module with `--tests=false` and run `make build`. `make test` and the CI test step cover only the root module.
