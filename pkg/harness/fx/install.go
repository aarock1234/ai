package fx

import (
	"strings"

	"github.com/aarock1234/ai/pkg/harness/acp"
)

// releaseVersion is the pinned fx release, the version of the libfx engine
// upstream pins (libfx 0.0.14).
const releaseVersion = "v0.0.14"

// releaseURL is fx's release download location.
const releaseURL = "https://releases.fx.sh"

// releaseArchive is the published release archive of one platform.
type releaseArchive struct {
	// pattern matches the "$(uname -s)-$(uname -m)" of the platform.
	pattern string

	// platform names the archive, fx-<platform>.tar.gz.
	platform string

	// sha256 is the published checksum of the archive.
	sha256 string
}

// releaseArchives are the fx v0.0.14 archives with the checksums published
// next to them as fx-<platform>.tar.gz.sha256.
var releaseArchives = []releaseArchive{
	{
		pattern:  "Linux-x86_64|Linux-amd64",
		platform: "linux-x86_64",
		sha256:   "731c1914fc2d497a10583f4416f8382c192e4897d6a0b4f713d6194455a8548b",
	},
	{
		pattern:  "Linux-aarch64|Linux-arm64",
		platform: "linux-aarch64",
		sha256:   "9f5cfbac1728fffddc2d5d874d1f7fe8d34f179e1bc86fb2e67a6e75074e53f3",
	},
	{
		pattern:  "Darwin-x86_64",
		platform: "macos-x86_64",
		sha256:   "6735101f94b1213278ade6109988f172493b863e8060b5152903d21467b95213",
	},
	{
		pattern:  "Darwin-arm64|Darwin-aarch64",
		platform: "macos-aarch64",
		sha256:   "619c6dccf4e8cd6290ec4a052d29c8724b2a4b150d893435b3de9086c21fb7b1",
	},
}

// implementationSource installs the pinned fx release into the private
// home's .local/bin, verifying the archive against its published checksum.
// It downloads the archive that fx's installer would select, without
// running the unpinned installer script.
func implementationSource() acp.InstallCommandSource {
	return acp.InstallCommandSource{
		Command: installCommand(),
	}
}

// installCommand returns the bash commands of the installation.
func installCommand() string {
	lines := []string{
		`case "$(uname -s)-$(uname -m)" in`,
	}

	for _, archive := range releaseArchives {
		lines = append(lines,
			"  "+archive.pattern+")",
			`    FX_PLATFORM="`+archive.platform+`"`,
			`    FX_SHA256="`+archive.sha256+`"`,
			"    ;;",
		)
	}

	lines = append(lines,
		"  *)",
		`    echo "fx does not support $(uname -s) $(uname -m)" >&2`,
		"    exit 1",
		"    ;;",
		"esac",
		"",
		`curl -fsSL "`+releaseURL+"/"+releaseVersion+`/fx-$FX_PLATFORM.tar.gz" -o fx.tar.gz`,
		"",
		"if command -v sha256sum >/dev/null 2>&1; then",
		`  echo "$FX_SHA256  fx.tar.gz" | sha256sum -c -`,
		"else",
		`  echo "$FX_SHA256  fx.tar.gz" | shasum -a 256 -c -`,
		"fi",
		"",
		"tar -xzf fx.tar.gz fx",
		`mv fx "$HOME/.local/bin/fx"`,
		`chmod +x "$HOME/.local/bin/fx"`,
		"rm fx.tar.gz",
	)

	return strings.Join(lines, "\n")
}
