Execute code-mode TypeScript in an isolated sandbox.

Put the full program in `js`; top-level `await`/`return` work. Return a JSON-serializable result.
Call host tools only as async `tools.name(input)`; await each or use `Promise.all` for independent calls.
{{if .Conversation}}Use exact names/types from the latest capability update.{{else}}Use exact names/types below.{{end}} `JSON.parse`/`JSON.stringify` are available.
Fetch: `fetch` is not available.

Tools:
{{if .Conversation}}The current host-tool API is provided in "Code mode capability update" user messages. Follow the latest catalog and ignore earlier catalogs.{{else}}{{.Catalog}}{{end}}
