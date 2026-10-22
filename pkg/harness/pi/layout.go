package pi

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"path"
	"regexp"
	"strings"

	"github.com/aarock1234/ai/pkg/harness"
)

// sessionFileNamePattern accepts the safe basenames of Pi session files.
var sessionFileNamePattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*\.jsonl?$`)

// launchIDPattern accepts the identifiers of relay state directories.
var launchIDPattern = regexp.MustCompile(`^[a-f0-9]{32}$`)

// skillsDir is where the adapter writes skills, relative to HOME. Pi
// discovers user-level Agent Skills there.
const skillsDir = ".agents/skills"

// layout locates the files of one session inside the sandbox.
type layout struct {
	homeDir    string
	workDir    string
	installDir string

	// privateDir holds the session's Pi state outside the working directory:
	// the agent directory, the session files, the extension configuration and
	// the relay state of each launch.
	privateDir string
}

// newLayout returns the layout of a session. The private directory is keyed
// by a hash of the session ID under HOME, as upstream, and must lie outside
// the working directory.
func newLayout(homeDir string, workDir string, sessionID string) (layout, error) {
	sum := sha256.Sum256([]byte(sessionID))
	privateDir := path.Join(homeDir, ".ai-sdk", "harness-pi", hex.EncodeToString(sum[:]))

	if isWithin(workDir, privateDir) {
		return layout{}, fmt.Errorf("pi private session directory %q must be outside the session working directory %q", privateDir, workDir)
	}

	return layout{
		homeDir:    homeDir,
		workDir:    workDir,
		installDir: path.Join(harness.StateDirectoryPath(homeDir), bootstrapDir),
		privateDir: privateDir,
	}, nil
}

// agentDir is Pi's agent directory (PI_CODING_AGENT_DIR), isolated per
// session.
func (l layout) agentDir() string {
	return path.Join(l.privateDir, "agent")
}

// sessionDir is where Pi stores its session files.
func (l layout) sessionDir() string {
	return path.Join(l.privateDir, "sessions")
}

// sessionFile returns the path of a session file.
func (l layout) sessionFile(name string) string {
	return path.Join(l.sessionDir(), name)
}

// extensionDir holds the application's Pi extensions.
func (l layout) extensionDir() string {
	return path.Join(l.privateDir, "extensions")
}

// extensionConfigFile is the configuration of the adapter's extension.
func (l layout) extensionConfigFile() string {
	return path.Join(l.privateDir, "extension-config.json")
}

// turnFile holds the instructions and application tools of the current
// turn.
func (l layout) turnFile() string {
	return path.Join(l.privateDir, "turn.json")
}

// skillsRoot is the harness skills directory.
func (l layout) skillsRoot() string {
	return path.Join(l.homeDir, skillsDir)
}

// relayStateDir is the relay state directory of one launch.
func (l layout) relayStateDir(launchID string) string {
	return path.Join(l.privateDir, "rpc", launchID)
}

// installFile returns a file of the installation.
func (l layout) installFile(name string) string {
	return path.Join(l.installDir, name)
}

// newLaunchID returns a random relay launch identifier.
func newLaunchID() string {
	random := make([]byte, 16)
	_, _ = rand.Read(random)

	return hex.EncodeToString(random)
}

// isWithin reports whether candidate is parent or lies below it.
func isWithin(parent string, candidate string) bool {
	parent = path.Clean(parent)
	candidate = path.Clean(candidate)

	return candidate == parent || strings.HasPrefix(candidate, strings.TrimSuffix(parent, "/")+"/")
}

// isSafeSessionFileName reports a session file basename without traversal.
func isSafeSessionFileName(name string) bool {
	return sessionFileNamePattern.MatchString(name)
}
