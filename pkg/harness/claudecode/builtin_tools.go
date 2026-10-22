package claudecode

import (
	_ "embed"
	"fmt"
	"slices"
	"sync"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
)

// builtinToolTable declares every native tool the Claude Code CLI can
// invoke, keyed by what the bridge reports as the tool name. The schemas are
// transcribed from upstream, which derives them from the pinned Claude Code
// executable. The generic MCP proxy tool is omitted: the bridge filters the
// application's MCP tools out, and other MCP servers report their own tool
// names.
//
//go:embed builtin_tools.json
var builtinToolTable []byte

// parsedBuiltinTools decodes the table once. The table is part of the
// package, so a failure is a programming error.
var parsedBuiltinTools = sync.OnceValue(func() []harness.BuiltinTool {
	tools, err := bridgeadapter.ParseBuiltinTools(builtinToolTable)
	if err != nil {
		panic(fmt.Sprintf("claude code built-in tools: %v", err))
	}

	return tools
})

// builtinTools returns a copy of the built-in tools.
func builtinTools() []harness.BuiltinTool {
	return slices.Clone(parsedBuiltinTools())
}
