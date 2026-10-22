package claudecode

import (
	_ "embed"
	"path"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
)

// bootstrapDir is where the recipe installs the bridge, relative to the
// harness state directory under the sandbox's HOME. Snapshot-capable
// sandboxes preserve the installed CLI and bridge there.
const bootstrapDir = ".harness-bootstrap/claude-code"

// The bridge that runs inside the sandbox is JavaScript for the sandbox's
// Node.js runtime. The assets directory holds the files upstream ships for
// it, copied verbatim from the dist/bridge directory of the published
// @ai-sdk/harness-claude-code 1.0.161 package, whose sources match the
// pinned upstream reference: the bundled bridge (index.mjs) and the bridge's
// package.json, pnpm-lock.yaml and pnpm-workspace.yaml. They are Copyright
// 2023 Vercel, Inc. and licensed under the Apache License, Version 2.0; see
// assets/LICENSE.
var (
	//go:embed assets/index.mjs
	bridgeScript string

	//go:embed assets/package.json
	bridgePackageJSON string

	//go:embed assets/pnpm-lock.yaml
	bridgeLockfile string

	//go:embed assets/pnpm-workspace.yaml
	bridgeWorkspace string
)

// bootstrapRecipe returns the recipe that installs the bridge with its
// locked dependencies, which include the Claude Code CLI.
func bootstrapRecipe() *harness.Bootstrap {
	return &harness.Bootstrap{
		HarnessID:    HarnessID,
		BootstrapDir: bootstrapDir,
		Files: []harness.BootstrapFile{
			{
				Path:    path.Join(bootstrapDir, "package.json"),
				Content: bridgePackageJSON,
			},
			{
				Path:    path.Join(bootstrapDir, "pnpm-lock.yaml"),
				Content: bridgeLockfile,
			},
			{
				Path:    path.Join(bootstrapDir, "pnpm-workspace.yaml"),
				Content: bridgeWorkspace,
			},
			{
				Path:    path.Join(bootstrapDir, bridgeadapter.LaunchScript),
				Content: bridgeScript,
			},
		},
		Commands: []harness.BootstrapCommand{
			{
				Command: "pnpm install --frozen-lockfile --store-dir .pnpm-store",
			},
			{
				Command: "./node_modules/.bin/claude --version",
			},
		},
	}
}
