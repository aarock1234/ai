package pi

import (
	_ "embed"
	"path"

	"github.com/aarock1234/ai/pkg/harness"
)

// bootstrapDir is where the recipe installs Pi, relative to the harness
// state directory under the sandbox's HOME. Snapshot-capable sandboxes
// preserve the installation there.
const bootstrapDir = ".harness-bootstrap/pi"

// Files of the bootstrap directory.
const (
	relayScript     = "relay.mjs"
	followScript    = "follow.mjs"
	extensionScript = "extension.ts"

	// piCLI is Pi's command-line entry point, relative to the bootstrap
	// directory.
	piCLI = "node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js"
)

// Pi runs inside the sandbox as a locked pnpm project that depends on the
// pinned @earendil-works/pi-coding-agent version upstream uses. Upstream
// embeds Pi in the host process and ships no sandbox files, so the assets
// are this adapter's own: the project's package.json and its pnpm lockfile,
// the RPC relay (relay.mjs), the outbox follower (follow.mjs) and the Pi
// extension (extension.ts).
var (
	//go:embed assets/package.json
	runtimePackageJSON string

	//go:embed assets/pnpm-lock.yaml
	runtimeLockfile string

	//go:embed assets/relay.mjs
	relaySource string

	//go:embed assets/follow.mjs
	followSource string

	//go:embed assets/extension.ts
	extensionSource string
)

// bootstrapRecipe returns the recipe that installs Pi with its locked
// dependencies next to the relay, the follower and the extension.
func bootstrapRecipe() *harness.Bootstrap {
	return &harness.Bootstrap{
		HarnessID:    HarnessID,
		BootstrapDir: bootstrapDir,
		Files: []harness.BootstrapFile{
			{
				Path:    path.Join(bootstrapDir, "package.json"),
				Content: runtimePackageJSON,
			},
			{
				Path:    path.Join(bootstrapDir, "pnpm-lock.yaml"),
				Content: runtimeLockfile,
			},
			{
				Path:    path.Join(bootstrapDir, relayScript),
				Content: relaySource,
			},
			{
				Path:    path.Join(bootstrapDir, followScript),
				Content: followSource,
			},
			{
				Path:    path.Join(bootstrapDir, extensionScript),
				Content: extensionSource,
			},
		},
		Commands: []harness.BootstrapCommand{
			{
				Command: "pnpm install --frozen-lockfile --store-dir .pnpm-store",
			},
		},
	}
}
