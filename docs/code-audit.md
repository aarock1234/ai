# Code audit and rewrite (October 2026)

## Findings

A read-only audit of the previous implementation (about 65k lines, mostly agent-written) covered every production package and the examples. The main findings:

- **Parallel implementations.** Anthropic had a `modern_*` implementation beside an older one; two stream decoders behaved differently, and the public decoder was obsolete.
- **Untyped wire handling.** Known JSON shapes were handled as `map[string]json.RawMessage`, re-decoded repeatedly, or edited as raw bytes; decode errors were silently ignored in dozens of places.
- **Duplicated plumbing.** The HTTP exchange was copied about 16 times across providers. The copies had drifted: multi-value headers were lost, and diagnostics and cancellation handling were inconsistent.
- **Contract sprawl.** `pkg/model` kept legacy fields beside their replacements, had five media shapes and five provider-data shapes, and spread validation and cloning over 15 files.
- **Over-built orchestration.** `pkg/ai` had about 180 files, 34 continuation entry points and about 5.5k lines of Go-only approval machinery beyond upstream.
- **Readability.** The audit found dense multi-clause conditions, shadowed package names (`context`, `copy`), compressed literals, 100-line functions and inconsistent errors.

## Resolution

The implementation was rewritten from scratch as an idiomatic port of the pinned upstream source rather than patched. The previous tree is preserved outside the repository for reference.

| Area | Result |
| --- | --- |
| `pkg/model` | Mirrors provider specification v4: typed per-role messages, sealed unions, optional pointers, typed provider values read with `model.ProviderValue`. |
| `internal/` | One owner per concern: `apicall` (HTTP exchange, header merging, `APICallError`, `Retry-After`), `apikey`, `sse`, `chatsse` (OpenAI-style chat SSE loop), `websocket` (RFC 6455 client), `download` (SSRF-safe downloads and validating client), `dataurl`, `origin`, `poll`, `partialjson` (port of `fixJson`), `schema`, `mediatype`, `toolnames`, `toolcalls`, `toolresult`, `reasoning`, `jsonobject`, `merge`, `optional`, `id`, `wirejson` (upstream JSON shapes), `awssigv4`, `awseventstream`, `livetranscription`, `transcriptionstream`, `geminilive`, `decisionwire`, `decisionlm`. |
| Providers | Each package is ported from upstream with typed wire structs and discriminated-union decoding, with consistent constructor, options and metadata naming. |
| `pkg/ai` | Upstream `generate-text`, `prompt`, tools, `embed`, media, `registry` and `agent`, plus `pkg/ai/middleware` and `pkg/ai/ui`. Streams are pull-based and single-consumer. |
| Review | A follow-up review fixed seven correctness issues: a post-stop `yield`, approval signatures that merged large integers, a transform data race, and others. It also aligned naming across providers and split the largest files. |

## Second review round

After the remaining roadmap phases landed, five independent reviewers re-read all production and example code by area, and a sixth pass read in full the files the first readers had only skimmed. Every reported finding was fixed. The six stream defects found in the original Anthropic implementation were separately verified as resolved against the pinned upstream source.

| Area | Main findings fixed |
| --- | --- |
| Credentials and security | A refresh token passed on the macOS command line; four non-atomic credential writers sharing a temp path; credential files briefly world-readable; OAuth refreshes following redirects; an unvalidated token endpoint; auth transports sending credentials to any host; SSRF gaps through IPv6 6to4 and Teredo addresses; unescaped IDs and API keys in URLs; SigV4 signing only POST bodies and sorting query parameters before encoding. |
| Concurrency and ownership | Bridge commands lost or reordered after reconnects; network writes under a shared mutex; unowned goroutines; process kill races; a race in `ChatStream.Close`; failed stream attempts left open during retry backoff; leaked batch result bodies; realtime state recording unsent tool output. |
| Correctness | Tool timeouts not surfacing as `*TimeoutError`; nil-pointer panics on missing URLs, messages and caller functions; unbounded allocations from decision scores; invalid JSON from `jsonobject`; WebSocket close-frame handling; infinite pagination; background jobs not cancelled on poll timeout; silently ignored options and decode errors. |
| Structure | One generic live-session implementation replacing four copies; shared JSON Lines, URL-path, data-URL, API-key, shell-quoting, polling and origin helpers; ACP reusing the shared bridge launcher; optional per-operation telemetry interfaces; consistent hook, sentinel, token-source and metadata naming across providers. |
| Readability | Long functions split, dense conditions named, shadowed names and compressed literals removed, untyped known shapes replaced with typed structs. |

## Verification

Final state, for each of the five modules (the core module, `otel/`, `codemode/`, `tui/` and `policyopawasm/`):

- `gofmt -l .` reports no files.
- `go build ./...` succeeds.
- `go vet` passes on every package's production files.
- The pinned golangci-lint v2.15.0 reports 0 issues on `run --tests=false ./...`.
- Cross-compiled builds pass for `linux/amd64`, `linux/arm64` and `darwin/arm64`.

Nothing has been exercised at runtime, and no live provider calls were made. The pre-rewrite `*_test.go` files, which no longer compiled against the new API, were deleted at the owner's request; the repository has no tests. See [roadmap](roadmap.md) and [feature parity](feature-parity.md) for remaining work.
