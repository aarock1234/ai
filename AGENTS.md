# Repository guidance

- Use the `go-style` skill and the shared coding philosophy. Compare relevant examples with `../go-template` when available; this is a published SDK, not an application template.
- `go.mod` defines the supported Go baseline. Preserve public imports under `pkg/`; use `internal/` for implementation details that consumers must not import.
- Keep normalized provider contracts in `pkg/model`, consumer orchestration in `pkg/ai`, and provider wire formats in their owning provider packages. Do not copy the template's application layers or specialized HTTP transport into the SDK.
- Expand every nonempty struct declaration, struct literal, and keyed map literal, including nested and one-field values. Review readability after formatting.
- Preserve caller cancellation, response-body ownership, error causes, content order, and provider-specific opaque data. Do not silently drop unsupported options or malformed required data.
- Never add, write, run, propose, or plan tests without an explicit request. `make check-static` builds, lints production code, and checks formatting; `make check` also runs tests and rewrites formatting.
- Prioritize coding in large, coherent batches. Do not repeat builds, lint, vet, module verification or cross-compilation after small edits or every iteration. Run relevant checks once when a substantial finishing task is ready; reserve full multi-platform checks for major phase completion/release, or a concrete platform-specific change.
- Use authorized subagents for independent implementation, with coordinator integration/review. Keep commentary, documentation updates and goal checkpoints concise; report pending verification honestly rather than running checks merely to produce a progress update.
- `make lint` and `make format-check` use the pinned Golangci-lint version in `Makefile`; keep CI and release pins aligned. Do not use automatic source fixes during an audit.
- Read `docs/feature-parity.md` and `docs/roadmap.md` before parity work. Implement features against the pinned upstream evidence, not historical unchecked plans or remembered TypeScript APIs.
- Preserve pre-existing changes. Keep logs lowercase except proper nouns, and keep prompt instructions in dedicated Markdown files when authoring or revising them.
