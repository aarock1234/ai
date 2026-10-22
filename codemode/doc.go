// Package codemode lets models write TypeScript that calls the
// application's tools, a port of the Vercel AI SDK @ai-sdk/code-mode
// package. It is a separate module, so that the core module does not
// depend on a JavaScript interpreter or a TypeScript transformer.
//
// NewTool returns an ai.Tool whose description declares the host tools as
// TypeScript signatures with call examples. The model writes a program in
// its js input; its TypeScript syntax is stripped with
// github.com/evanw/esbuild, and the program runs in a fresh
// github.com/dop251/goja runtime, calls host tools as tools.name(input),
// and returns a JSON-serializable value, which becomes the tool's output:
//
//	codeMode := codemode.NewTool("code_mode", []ai.Tool{getInventory, getDemand}, codemode.Options{})
//
//	result, err := ai.GenerateText(ctx, ai.GenerateTextOptions{
//		Model: languageModel,
//		Tools: []ai.Tool{codeMode},
//		Prompt: ai.Prompt{
//			Prompt: "Compare inventory and demand for product sku_123.",
//		},
//	})
//
// NewCallerTool instead binds the host tools through
// ai.GenerateTextOptions.ToolCallers, and Run executes a program directly.
//
// Values cross the sandbox boundary as JSON only. Host tool inputs are
// validated against the tools' input schemas, tools that need approval
// follow Options.Approval, and ExecutionPolicy bounds the duration, call
// depth, source, result, console, tool input and output sizes, and the
// number of host tool calls. SetMaxWorkers caps the invocations that
// execute at the same time in the process. A failed host tool call rejects
// the program's promise with an error that carries the failure's message
// and code; when the program does not catch it, Run returns the original
// Go error.
//
// A host tool may interrupt the invocation, for example to ask for
// approval or authentication, by returning RequestInterrupt's error. The
// invocation then stops with an Interrupt carrying a signed Continuation.
// ContinueInterrupt and ContinueApproval resume it: the program runs again
// from the start, host tool calls that settled are replayed from the
// continuation in the order they settled instead of running again, the
// program's clock and Math.random repeat the values they returned, and the
// interrupted calls receive their resolutions. Continuations use upstream's
// format, signature and replay semantics, so a continuation created by
// upstream resumes here and the other way around, given the same signing
// key.
//
// Go adaptations: cancellation and the timeout come from the context and
// ExecutionPolicy.Timeout. Upstream's application context is the context
// passed to host tools, and a resumed tool reads its resolution with
// ResumedInterruptFromContext. The interpreter cannot bound memory, so
// upstream's memory limit is not available, the stack limit is a call
// depth, and the default worker cap is upstream's upper bound rather than
// one derived from the available memory. esbuild also compiles TypeScript
// syntax that upstream's type stripping rejects, such as enums. Outputs of
// tools without an ai.Tool.OutputSchema are declared as unknown.
package codemode
