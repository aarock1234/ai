package codex

import (
	_ "embed"
	"fmt"
	"slices"
	"sync"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
)

// builtinToolTable declares every native tool the Codex app-server can
// invoke as a model-callable tool, keyed by what the bridge reports as the
// tool name. Other native operations, such as todo planning, surface as side
// effects rather than tools.
//
//go:embed builtin_tools.json
var builtinToolTable []byte

// parsedBuiltinTools decodes the table once. The table is part of the
// package, so a failure is a programming error.
var parsedBuiltinTools = sync.OnceValue(func() []harness.BuiltinTool {
	tools, err := bridgeadapter.ParseBuiltinTools(builtinToolTable)
	if err != nil {
		panic(fmt.Sprintf("codex built-in tools: %v", err))
	}

	return tools
})

// builtinTools returns a copy of the built-in tools.
func builtinTools() []harness.BuiltinTool {
	return slices.Clone(parsedBuiltinTools())
}
