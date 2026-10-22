package codex

import (
	_ "embed"
	"path"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
)

// bootstrapDir is where the recipe installs the bridge, relative to the
// harness state directory under the sandbox's HOME. Snapshot-capable
// sandboxes preserve the installed CLI and bridge there.
const bootstrapDir = ".harness-bootstrap/codex"

// The bridge that runs inside the sandbox is JavaScript for the sandbox's
// Node.js runtime. The assets directory holds the files upstream ships for
// it, copied verbatim from the dist/bridge directory of the published
// @ai-sdk/harness-codex 1.0.159 package, whose sources match the pinned
// upstream reference: the bundled bridge (index.mjs) and the bridge's
// package.json and pnpm-lock.yaml. They are Copyright 2023 Vercel, Inc. and
// licensed under the Apache License, Version 2.0; see assets/LICENSE.
var (
	//go:embed assets/index.mjs
	bridgeScript string

	//go:embed assets/package.json
	bridgePackageJSON string

	//go:embed assets/pnpm-lock.yaml
	bridgeLockfile string
)

// bootstrapRecipe returns the recipe that installs the bridge with its
// locked dependencies, which include the Codex CLI. The bridge lies directly
// in the bootstrap directory because it locates the CLI relative to itself.
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
				Path:    path.Join(bootstrapDir, bridgeadapter.LaunchScript),
				Content: bridgeScript,
			},
		},
		Commands: []harness.BootstrapCommand{
			{
				Command: "pnpm install --frozen-lockfile --store-dir .pnpm-store",
			},
		},
	}
}
