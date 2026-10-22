package harness

import (
	"context"
	"fmt"
	"path"
	"strings"

	"github.com/aarock1234/ai/pkg/sandbox"
)

// stateDirectoryName is the directory, under the sandbox's HOME, that holds
// every piece of state the harness machinery generates.
const stateDirectoryName = ".ai-sdk-harness"

// NetworkSession is a sandbox session with the infrastructure surface a
// harness needs: port resolution, lifecycle and network policy. Sandbox
// providers such as sandbox/vercel and sandbox/local return one; the agent
// keeps it for the lifetime of a harness session.
//
// Code that should only touch files and processes, such as application
// tools, receives the reduced view from Restricted instead, so it cannot
// stop the sandbox or change its network access. Optional capabilities are
// the NetworkPolicySetter, RequestTransformationSetter,
// RequestTransformationAdder and PortSetter interfaces.
type NetworkSession interface {
	sandbox.Session

	// ID identifies the sandbox. Persist it separately from the harness
	// session ID and lifecycle state to reattach across processes.
	ID() string

	// DefaultWorkingDirectory is the absolute directory relative commands run
	// in, read from the live sandbox. The agent composes session working
	// directories under it.
	DefaultWorkingDirectory() string

	// Ports are the ports the sandbox exposes.
	Ports() []int

	// PortEndpoint resolves the connection details of an exposed port, for
	// example the WebSocket endpoint of an in-sandbox bridge.
	PortEndpoint(ctx context.Context, options PortEndpointOptions) (PortEndpoint, error)

	// Stop stops the sandbox. It is idempotent.
	Stop(ctx context.Context) error

	// Destroy stops the sandbox and releases its backing resources. It
	// handles both running and stopped sandboxes.
	Destroy(ctx context.Context) error

	// Restricted returns a view of the same sandbox limited to files and
	// processes. The view must not implement NetworkSession.
	Restricted() sandbox.Session
}

// NetworkPolicySetter is implemented by network sessions that can enforce an
// outbound network policy.
type NetworkPolicySetter interface {
	SetNetworkPolicy(ctx context.Context, policy NetworkPolicy) error
}

// RequestTransformationSetter is implemented by network sessions that can
// rewrite outbound requests outside the sandbox security boundary, for
// example to inject credentials. Setting replaces the complete
// transformation set; adapters should normally add instead.
type RequestTransformationSetter interface {
	SetRequestTransformations(ctx context.Context, transformations []RequestTransformation) error
}

// RequestTransformationAdder is implemented by network sessions that can add
// outbound request transformations without replacing the rules they already
// manage.
type RequestTransformationAdder interface {
	AddRequestTransformations(ctx context.Context, transformations []RequestTransformation) error
}

// PortSetter is implemented by network sessions that can change their
// exposed ports. Ports omitted from the list are no longer exposed.
type PortSetter interface {
	SetPorts(ctx context.Context, ports []int) error
}

// PortProtocol is the protocol of a port endpoint URL.
type PortProtocol string

// Port protocols. The empty protocol is PortHTTPS.
const (
	PortHTTP      PortProtocol = "http"
	PortHTTPS     PortProtocol = "https"
	PortWebSocket PortProtocol = "ws"
)

// PortEndpointOptions configure NetworkSession.PortEndpoint.
type PortEndpointOptions struct {
	Port     int
	Protocol PortProtocol
}

// PortEndpoint is the connection details of an exposed port. Headers are
// scoped to URL and must be sent when connecting.
type PortEndpoint struct {
	URL     string
	Headers map[string]string
}

// NetworkPolicyMode selects a network policy.
type NetworkPolicyMode string

// Network policy modes.
const (
	NetworkAllowAll NetworkPolicyMode = "allow-all"
	NetworkDenyAll  NetworkPolicyMode = "deny-all"
	NetworkCustom   NetworkPolicyMode = "custom"
)

// NetworkPolicy is an outbound network policy. A custom policy is an
// allow-list: reachable hosts are the union of AllowedHosts and
// AllowedCIDRs, and DeniedCIDRs take precedence over both, for example to
// block cloud metadata addresses. A custom policy needs at least one allowed
// host or CIDR.
type NetworkPolicy struct {
	Mode NetworkPolicyMode

	AllowedHosts []string
	AllowedCIDRs []string
	DeniedCIDRs  []string
}

// RequestTransformation rewrites matching outbound HTTPS requests outside
// the sandbox. Credential values belong in Transform.Headers, while the
// sandbox process sees only a placeholder; implementations overwrite the
// headers after the request leaves the sandbox. Several rules, including
// several for one host, can be installed at once.
type RequestTransformation struct {
	Match     RequestMatch     `json:"match"`
	Transform RequestTransform `json:"transform"`
}

// RequestMatch selects the requests a transformation applies to.
type RequestMatch struct {
	Host        string            `json:"host"`
	Path        *Matcher          `json:"path,omitempty"`
	Method      []string          `json:"method,omitempty"`
	QueryString []KeyValueMatcher `json:"queryString,omitempty"`
	Headers     []KeyValueMatcher `json:"headers,omitempty"`
}

// Matcher matches a string exactly, by prefix or by regular expression. Set
// exactly one field.
type Matcher struct {
	Exact      string `json:"exact,omitempty"`
	StartsWith string `json:"startsWith,omitempty"`
	Regex      string `json:"regex,omitempty"`
}

// KeyValueMatcher matches a query parameter or header by key and value.
type KeyValueMatcher struct {
	Key   *Matcher `json:"key,omitempty"`
	Value *Matcher `json:"value,omitempty"`
}

// RequestTransform is the rewrite of a matching request.
type RequestTransform struct {
	// Headers overwrite the request headers of the same names.
	Headers map[string]string `json:"headers"`
}

// SandboxTemplate prepares reusable sandbox state, such as a snapshot, for
// harness sessions. Sandbox providers that support snapshots key the
// snapshot by Identity and call Prepare once per identity; others call
// Prepare on every creation.
type SandboxTemplate struct {
	Identity string

	// Prepare applies the template to a freshly created sandbox.
	Prepare func(ctx context.Context, session sandbox.Session) error
}

// SandboxSessionCreateOptions are the provider-independent options of a
// sandbox session creator. Providers embed them in their own options.
type SandboxSessionCreateOptions struct {
	// SandboxID names the new sandbox. Creation never resumes an existing
	// sandbox: an existing sandbox with the same ID is a conflict.
	SandboxID string

	// Template is applied to the new sandbox. Nil applies none.
	Template *SandboxTemplate
}

// SandboxSessionResumeOptions are the provider-independent options of a
// sandbox session resumer.
type SandboxSessionResumeOptions struct {
	// SandboxID identifies the existing sandbox. Resuming never creates a
	// sandbox when it cannot be found.
	SandboxID string
}

// StateDirectoryPath returns the fixed directory for harness-generated state
// (bootstrap files, markers and session data) under the sandbox's HOME. It
// never lies under the default working directory. The agent also calls it
// with a symbolic "$HOME" when hashing bootstrap recipes, so changing it
// invalidates existing templates.
func StateDirectoryPath(sandboxHomeDir string) string {
	return path.Join(sandboxHomeDir, stateDirectoryName)
}

// SessionDataDirectoryPath returns the per-session state directory under the
// harness state directory. The session ID occupies one encoded path
// segment, so separators and parent references cannot escape it.
func SessionDataDirectoryPath(stateDirectory string, sessionID string) string {
	return path.Join(stateDirectory, ".agent-runs", EncodePathSegment(sessionID))
}

// EncodePathSegment encodes value as one path segment, as the JavaScript
// encodeURIComponent function does, and additionally encodes the empty
// string, "." and "..".
func EncodePathSegment(value string) string {
	encoded := encodeURIComponent(value)

	switch encoded {
	case "":
		return "%"
	case ".":
		return "%2E"
	case "..":
		return "%2E%2E"
	default:
		return encoded
	}
}

// RestrictedSession returns the files-and-processes view of session: the
// Restricted view of a NetworkSession, and session itself otherwise.
func RestrictedSession(session sandbox.Session) sandbox.Session {
	if network, ok := session.(NetworkSession); ok {
		return network.Restricted()
	}

	return session
}

// encodeURIComponent escapes every byte except the characters JavaScript's
// encodeURIComponent leaves unescaped.
func encodeURIComponent(value string) string {
	var encoded strings.Builder

	for index := 0; index < len(value); index++ {
		character := value[index]
		if isURIComponentSafe(character) {
			encoded.WriteByte(character)

			continue
		}

		fmt.Fprintf(&encoded, "%%%02X", character)
	}

	return encoded.String()
}

// isURIComponentSafe reports whether encodeURIComponent keeps character.
func isURIComponentSafe(character byte) bool {
	switch {
	case 'a' <= character && character <= 'z',
		'A' <= character && character <= 'Z',
		'0' <= character && character <= '9':
		return true
	}

	return strings.IndexByte("-_.!~*'()", character) >= 0
}
