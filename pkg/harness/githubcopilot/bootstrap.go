package githubcopilot

import (
	_ "embed"

	"github.com/aarock1234/ai/pkg/harness/acp"
)

// The GitHub Copilot CLI is installed inside the sandbox as a locked pnpm
// project. The assets directory holds the project files upstream inlines
// into its bundle, copied verbatim from the src/bridge directory of the
// published @ai-sdk/harness-github-copilot 1.0.53 package, whose sources
// match the pinned upstream reference: package.json, pnpm-lock.yaml and
// pnpm-workspace.yaml. They are Copyright 2023 Vercel, Inc. and licensed
// under the Apache License, Version 2.0; see assets/LICENSE.
var (
	//go:embed assets/package.json
	implementationPackageJSON string

	//go:embed assets/pnpm-lock.yaml
	implementationLockfile string

	//go:embed assets/pnpm-workspace.yaml
	implementationWorkspace string
)

// implementationSource installs the pinned GitHub Copilot CLI.
func implementationSource() acp.NPMLockedSource {
	return acp.NPMLockedSource{
		PackageJSON:       implementationPackageJSON,
		PNPMLockYAML:      implementationLockfile,
		PNPMWorkspaceYAML: implementationWorkspace,
	}
}
