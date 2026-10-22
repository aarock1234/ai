package opencode

import (
	_ "embed"
	"fmt"
	"slices"
	"sync"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
)

// builtinToolTable declares every native tool OpenCode can invoke, keyed by
// what the bridge reports as the tool name, in upstream order. The input
// schemas are loose objects: OpenCode accepts fields the table does not
// list.
//
//go:embed builtin_tools.json
var builtinToolTable []byte

// parsedBuiltinTools decodes the table once. The table is part of the
// package, so a failure is a programming error.
var parsedBuiltinTools = sync.OnceValue(func() []harness.BuiltinTool {
	tools, err := bridgeadapter.ParseBuiltinTools(builtinToolTable)
	if err != nil {
		panic(fmt.Sprintf("opencode built-in tools: %v", err))
	}

	return tools
})

// builtinTools returns a copy of the built-in tools.
func builtinTools() []harness.BuiltinTool {
	return slices.Clone(parsedBuiltinTools())
}

// validateBuiltinToolFiltering rejects filtering that keeps only one of
// write and edit: OpenCode controls both with its single edit permission.
func validateBuiltinToolFiltering(filtering *harness.BuiltinToolFiltering) error {
	if filtering == nil {
		return nil
	}

	writeActive := harness.IsBuiltinToolIncluded(string(harness.ToolWrite), filtering)
	editActive := harness.IsBuiltinToolIncluded(string(harness.ToolEdit), filtering)

	if writeActive != editActive {
		return unsupported("OpenCode built-in filtering requires 'write' and 'edit' to either both be active or both be inactive.")
	}

	return nil
}

// inactiveBuiltinToolNames returns the built-in tools the filtering removes,
// which the bridge disables in OpenCode. Nil filtering returns nil, so that
// the start command omits the list.
func inactiveBuiltinToolNames(filtering *harness.BuiltinToolFiltering) []string {
	if filtering == nil {
		return nil
	}

	inactive := []string{}

	for _, tool := range parsedBuiltinTools() {
		if !harness.IsBuiltinToolIncluded(tool.Name, filtering) {
			inactive = append(inactive, tool.Name)
		}
	}

	return inactive
}
