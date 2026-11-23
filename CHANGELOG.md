# Changelog

## v0.7.0

- A tool that panics is reported as a tool error with a `*ToolPanicError` instead of crashing the process.
- An empty `ToolApprovalSecret` is rejected. Leave it nil to disable signing.
- Tool approval signatures treat `10.0` and `10` as the same number, so approvals survive a JavaScript client. Approvals still pending from an earlier version may need to be requested again.
- Connection resets, refused connections, DNS failures and truncated responses are retried.
- API keys are no longer sent when a provider redirects to another origin.
- Connection URLs in WebSocket errors no longer include their query string.
- Gateway streams skip part types they do not know instead of ending.
- OpenAI-compatible streams always end with a finish part.
- Structured output streaming no longer panics on text cut in the middle of a character.

## v0.6.1

- Stopping a Vercel sandbox sends a JSON body.

## v0.6.0

- OpenRouter decision models call OpenRouter's Decisions API.
- OpenRouter supports automatic prompt caching with `CacheControl`.
- OpenRouter supports sticky sessions with `SessionID`.

## v0.5.0

- OpenRouter has decision models for `Decide`.
- `examples/decide` shows decisions with OpenAI and OpenRouter.

## v0.4.0

- The SDK was rewritten. See the [migration guide](docs/migration.md).
