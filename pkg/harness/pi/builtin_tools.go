package pi

import (
	_ "embed"
	"fmt"
	"slices"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
)

// builtinToolTable declares Pi's built-in tools with the upstream schemas,
// in upstream order. The adapter's Pi extension registers them in place of
// Pi's native declarations and delegates to the native implementations.
//
//go:embed builtin_tools.json
var builtinToolTable []byte

// builtinTools are the decoded built-in tools. The table is part of the
// package, so a failure is a programming error.
var builtinTools = func() []harness.BuiltinTool {
	tools, err := bridgeadapter.ParseBuiltinTools(builtinToolTable)
	if err != nil {
		panic(fmt.Sprintf("pi built-in tools: %v", err))
	}

	return tools
}()

// nativeBuiltinToolNames are Pi's native names of the built-in tools, in
// upstream order.
var nativeBuiltinToolNames = []string{
	"read",
	"write",
	"edit",
	"bash",
	"grep",
	"find",
	"ls",
}

// nativeToCommonName maps the native tools Pi reports under another name.
var nativeToCommonName = map[string]string{
	"find": "glob",
}

// commonName returns the public name of a native built-in tool.
func commonName(nativeName string) string {
	if name, ok := nativeToCommonName[nativeName]; ok {
		return name
	}

	return nativeName
}

// activeBuiltinToolNames returns the native names of the built-in tools the
// filtering keeps, in upstream order.
func activeBuiltinToolNames(filtering *harness.BuiltinToolFiltering) []string {
	return slices.DeleteFunc(slices.Clone(nativeBuiltinToolNames), func(nativeName string) bool {
		return !harness.IsBuiltinToolIncluded(commonName(nativeName), filtering)
	})
}
