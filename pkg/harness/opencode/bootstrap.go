package opencode

import (
	_ "embed"
	"path"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
)

// bootstrapDir is where the recipe installs the bridge, relative to the
// harness state directory under the sandbox's HOME. Snapshot-capable
// sandboxes preserve the installed CLI and bridge there.
const bootstrapDir = ".harness-bootstrap/opencode"

// The bridge that runs inside the sandbox is JavaScript for the sandbox's
// Node.js runtime. The assets directory holds the files upstream ships for
// it, copied verbatim from the dist/bridge directory of the published
// @ai-sdk/harness-opencode 1.0.159 package, whose sources match the pinned
// upstream reference: the bundled bridge (index.mjs), the host tool MCP
// server (host-tool-mcp.mjs) and the bridge's package.json, pnpm-lock.yaml
// and pnpm-workspace.yaml. They are Copyright 2023 Vercel, Inc. and licensed
// under the Apache License, Version 2.0; see assets/LICENSE.
var (
	//go:embed assets/index.mjs
	bridgeScript string

	//go:embed assets/host-tool-mcp.mjs
	hostToolMCPScript string

	//go:embed assets/package.json
	bridgePackageJSON string

	//go:embed assets/pnpm-lock.yaml
	bridgeLockfile string

	//go:embed assets/pnpm-workspace.yaml
	bridgeWorkspace string
)

// bootstrapRecipe returns the recipe that installs the bridge with its
// locked dependencies, which include the OpenCode CLI, and checks that the
// CLI runs.
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
			{
				Path:    path.Join(bootstrapDir, "host-tool-mcp.mjs"),
				Content: hostToolMCPScript,
			},
		},
		Commands: []harness.BootstrapCommand{
			{
				Command: "pnpm install --frozen-lockfile --store-dir .pnpm-store",
			},
			{
				Command: "./node_modules/.bin/opencode --version",
			},
		},
	}
}

// bridgeArguments returns the arguments upstream's command line passes the
// OpenCode bridge after the shared ones: the bootstrap directory, where the
// bridge finds the OpenCode CLI and the host tool MCP server, and the skills
// directory under the sandbox home that the host writes skills to.
func bridgeArguments(launch bridgeadapter.LaunchContext) []string {
	return []string{
		"--bootstrap-dir",
		launch.BootstrapDir,
		"--skills-dir",
		path.Join(launch.HomeDir, ".agents", "skills"),
	}
}
