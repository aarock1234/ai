package acp

import (
	_ "embed"
	"path"

	"github.com/aarock1234/ai/pkg/harness"
)

// The bridge that runs inside the sandbox is JavaScript for the sandbox's
// Node.js runtime. The assets directory holds the files upstream ships for
// it, copied verbatim from the dist/bridge directory of the published
// @ai-sdk/harness-acp 1.0.96 package, whose sources match the pinned
// upstream reference: the bundled bridge (index.mjs), the host tool MCP
// server (host-tool-mcp.mjs) and the bridge's package.json and
// pnpm-lock.yaml. They are Copyright 2023 Vercel, Inc. and licensed under
// the Apache License, Version 2.0; see assets/LICENSE.
var (
	//go:embed assets/index.mjs
	bridgeScript string

	//go:embed assets/host-tool-mcp.mjs
	hostToolMCPScript string

	//go:embed assets/package.json
	bridgePackageJSON string

	//go:embed assets/pnpm-lock.yaml
	bridgeLockfile string
)

// Bootstrap file names below the bootstrap directory.
const (
	bridgeFileName          = "bridge.mjs"
	implementationDirectory = "implementation"
)

// bootstrapDirectory returns the bootstrap directory of a harness, relative
// to the harness state directory.
func bootstrapDirectory(harnessID string) string {
	return path.Join(".harness-bootstrap", harnessID)
}

// newBootstrap returns the bootstrap recipe: the bridge with its locked
// dependencies and the implementation with its installation.
func newBootstrap(harnessID string, source implementation) (harness.Bootstrap, error) {
	bootstrapDir := bootstrapDirectory(harnessID)

	descriptor, err := source.descriptor()
	if err != nil {
		return harness.Bootstrap{}, err
	}

	files := []harness.BootstrapFile{
		{
			Path:    path.Join(bootstrapDir, "package.json"),
			Content: bridgePackageJSON,
		},
		{
			Path:    path.Join(bootstrapDir, "pnpm-lock.yaml"),
			Content: bridgeLockfile,
		},
		{
			Path:    path.Join(bootstrapDir, bridgeFileName),
			Content: bridgeScript,
		},
		{
			Path:    path.Join(bootstrapDir, "host-tool-mcp.mjs"),
			Content: hostToolMCPScript,
		},
		{
			Path:    path.Join(bootstrapDir, implementationDirectory, "implementation.json"),
			Content: descriptor,
		},
	}

	implementationFiles, err := implementationBootstrapFiles(path.Join(bootstrapDir, implementationDirectory), source)
	if err != nil {
		return harness.Bootstrap{}, err
	}

	return harness.Bootstrap{
		HarnessID:    harnessID,
		BootstrapDir: bootstrapDir,
		Files:        append(files, implementationFiles...),
		Commands: []harness.BootstrapCommand{
			{
				Command: "pnpm install --frozen-lockfile --store-dir .pnpm-store",
			},
			{
				Command: source.installCommand(implementationDirectory, "../.pnpm-store"),
			},
		},
	}, nil
}

// implementationBootstrapFiles returns the files that install the
// implementation: its manifest, lockfile and workspace file, or its install
// script.
func implementationBootstrapFiles(directory string, source implementation) ([]harness.BootstrapFile, error) {
	var files []harness.BootstrapFile

	manifest, ok, err := source.manifest()
	if err != nil {
		return nil, err
	}

	if ok {
		files = append(files, harness.BootstrapFile{
			Path:    path.Join(directory, "package.json"),
			Content: manifest,
		})
	}

	if locked, ok := source.lockedSource(); ok {
		files = append(files, harness.BootstrapFile{
			Path:    path.Join(directory, "pnpm-lock.yaml"),
			Content: locked.PNPMLockYAML,
		})

		if locked.PNPMWorkspaceYAML != "" {
			files = append(files, harness.BootstrapFile{
				Path:    path.Join(directory, "pnpm-workspace.yaml"),
				Content: locked.PNPMWorkspaceYAML,
			})
		}
	}

	if script, ok := source.installScript(); ok {
		files = append(files, harness.BootstrapFile{
			Path:    path.Join(directory, "install.sh"),
			Content: script,
		})
	}

	return files, nil
}
