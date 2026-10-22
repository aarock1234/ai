package deepagents

import (
	_ "embed"
	"fmt"
	"slices"
	"sync"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
)

// builtinToolTable declares every model-callable Deep Agents tool, keyed by
// what the bridge reports as the tool name: the common name when there is
// one, the native name otherwise. Every tool must be listed, or a call of
// an unlisted tool fails as an unknown tool. The schemas are transcribed
// from upstream.
//
//go:embed builtin_tools.json
var builtinToolTable []byte

// parsedBuiltinTools decodes the table once. The table is part of the
// package, so a failure is a programming error.
var parsedBuiltinTools = sync.OnceValue(func() []harness.BuiltinTool {
	tools, err := bridgeadapter.ParseBuiltinTools(builtinToolTable)
	if err != nil {
		panic(fmt.Sprintf("deep agents built-in tools: %v", err))
	}

	return tools
})

// builtinTools returns a copy of the built-in tools.
func builtinTools() []harness.BuiltinTool {
	return slices.Clone(parsedBuiltinTools())
}
