// Package vercel is a sandbox session backed by Vercel Sandbox, ephemeral
// Linux microVMs, the Go port of upstream @ai-sdk/sandbox-vercel. It talks
// to the Vercel Sandbox REST API directly, in place of the @vercel/sandbox
// SDK the upstream package wraps.
//
// Create returns a harness.NetworkSession: pass it to
// harness.Agent.CreateSession, or pass its Restricted view to tools such as
// tools/bash. Pass the agent's sandbox template to Create to reuse a
// prepared snapshot across sessions.
//
// Authentication uses explicit Credentials, a Vercel OIDC token from
// VERCEL_OIDC_TOKEN_FILE or VERCEL_OIDC_TOKEN, refreshed through the linked
// project like @vercel/oidc, or VERCEL_TOKEN with VERCEL_TEAM_ID and
// VERCEL_PROJECT_ID. For local development, the DevelopmentSignIn
// connection option falls back to DevelopmentCredentials, and SignIn runs
// the Vercel CLI's device sign-in with a caller-supplied prompt.
//
// Go adaptation: the deprecated sandbox provider (createVercelSandbox) is
// not ported.
package vercel

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/aarock1234/ai/pkg/harness"
)

// providerID identifies Vercel Sandbox in errors.
const providerID = "vercel-sandbox"

// Creation defaults. The 30-minute timeout replaces the API's 5-minute
// default, which is too short for multi-step agent workflows.
const (
	defaultTimeout = 30 * time.Minute
	defaultRuntime = "node24"
)

// SourceType selects the content a sandbox starts from.
type SourceType string

// Source types.
const (
	SourceGit      SourceType = "git"
	SourceTarball  SourceType = "tarball"
	SourceSnapshot SourceType = "snapshot"
)

// Source is the content a sandbox starts from.
type Source struct {
	Type SourceType `json:"type"`

	// URL is the repository or tarball URL.
	URL string `json:"url,omitempty"`

	// Depth, Revision, Username and Password configure git sources.
	Depth    int    `json:"depth,omitempty"`
	Revision string `json:"revision,omitempty"`
	Username string `json:"username,omitempty"`
	Password string `json:"password,omitempty"`

	// SnapshotID selects the snapshot of snapshot sources.
	SnapshotID string `json:"snapshotId,omitempty"`
}

// ConnectionOptions configure how the package reaches the API.
type ConnectionOptions struct {
	Credentials

	// BaseURL is the API base URL. Empty selects https://vercel.com/api.
	BaseURL string

	// HTTPClient sends the requests. Nil uses http.DefaultClient.
	HTTPClient *http.Client

	// DevelopmentSignIn enables the development fallback of
	// @vercel/sandbox: when no OIDC token can be obtained, the credentials
	// come from DevelopmentCredentials, which signs in through its Prompt
	// when the Vercel CLI has no stored access token. Upstream enables the
	// fallback in interactive terminals outside production and CI; set it
	// only in local development. Nil reports the authentication error.
	DevelopmentSignIn *DeviceSignIn
}

// Options configure Create.
type Options struct {
	harness.SandboxSessionCreateOptions
	ConnectionOptions

	// Runtime selects a legacy runtime image such as "node24". Image selects
	// a container image. Set at most one; when neither is set and Source is
	// not a snapshot, the runtime is "node24".
	Runtime string
	Image   string

	Source *Source

	// Ports are exposed through public URLs.
	Ports []int

	// Timeout is the sandbox's lifetime. Zero selects 30 minutes.
	Timeout time.Duration

	// VCPUs is the number of virtual CPUs. Zero uses the API default.
	VCPUs int

	// Persistent keeps the sandbox's filesystem across sessions. Nil uses the
	// API default.
	Persistent *bool

	// NetworkPolicy is the initial outbound network policy. Nil allows all.
	NetworkPolicy *harness.NetworkPolicy

	Env    map[string]string
	Tags   map[string]string
	Region string

	// SnapshotExpiration is the lifetime of snapshots, including template
	// snapshots. Nil uses the API default; zero never expires.
	SnapshotExpiration *time.Duration
}

// ResumeOptions configure Resume.
type ResumeOptions struct {
	harness.SandboxSessionResumeOptions
	ConnectionOptions
}

// Create creates a sandbox. With a template, it reuses the template's
// snapshot, creating and preparing it once per template identity, and
// starts the sandbox from it. Creation never resumes an existing sandbox; a
// SandboxID that names an existing sandbox is a conflict.
func Create(ctx context.Context, options Options) (*Session, error) {
	if options.Runtime != "" && options.Image != "" {
		return nil, errors.New("create vercel sandbox: runtime and image cannot be used together")
	}

	client, err := newClient(ctx, options.ConnectionOptions)
	if err != nil {
		return nil, err
	}

	base, err := baseCreateRequest(client, options)
	if err != nil {
		return nil, err
	}

	if options.Template == nil {
		base.Name = options.SandboxID

		state, err := client.createSandbox(ctx, base)
		if err != nil {
			return nil, fmt.Errorf("create vercel sandbox: %w", err)
		}

		return newSession(client, state), nil
	}

	return createFromTemplate(ctx, templateCreation{
		client:    client,
		base:      base,
		template:  *options.Template,
		sandboxID: options.SandboxID,
	})
}

// Resume reattaches to an existing sandbox, resuming its session when it
// stopped. It never creates a sandbox.
func Resume(ctx context.Context, options ResumeOptions) (*Session, error) {
	if options.SandboxID == "" {
		return nil, errors.New("resume vercel sandbox: sandbox id is required")
	}

	client, err := newClient(ctx, options.ConnectionOptions)
	if err != nil {
		return nil, err
	}

	state, err := client.getSandbox(ctx, options.SandboxID, true)
	if err != nil {
		return nil, fmt.Errorf("resume vercel sandbox %s: %w", options.SandboxID, err)
	}

	return newSession(client, state), nil
}

// newClient resolves the credentials, refreshing an expired OIDC token, and
// returns an API client.
func newClient(ctx context.Context, options ConnectionOptions) (*apiClient, error) {
	httpClient := options.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}

	credentials, err := resolveCredentials(ctx, options, httpClient)
	if err != nil {
		return nil, err
	}

	return newAPIClient(credentials, options.BaseURL, httpClient), nil
}

// baseCreateRequest converts the options into a creation request with the
// default runtime and timeout.
func baseCreateRequest(client *apiClient, options Options) (createSandboxRequest, error) {
	timeout := options.Timeout
	if timeout == 0 {
		timeout = defaultTimeout
	}

	ports := options.Ports
	if ports == nil {
		ports = []int{}
	}

	request := createSandboxRequest{
		ProjectID:  client.credentials.ProjectID,
		Ports:      ports,
		Source:     options.Source,
		Timeout:    timeout.Milliseconds(),
		Runtime:    options.Runtime,
		Image:      options.Image,
		Persistent: options.Persistent,
		Env:        options.Env,
		Tags:       options.Tags,
		Region:     options.Region,
	}

	isSnapshotSource := options.Source != nil && options.Source.Type == SourceSnapshot
	if request.Runtime == "" && request.Image == "" && !isSnapshotSource {
		request.Runtime = defaultRuntime
	}

	if options.VCPUs > 0 {
		request.Resources = &resources{
			VCPUs: options.VCPUs,
		}
	}

	if options.SnapshotExpiration != nil {
		expiration := options.SnapshotExpiration.Milliseconds()
		request.SnapshotExpiration = &expiration
	}

	if options.NetworkPolicy != nil {
		access, err := toAccessPolicy(*options.NetworkPolicy)
		if err != nil {
			return createSandboxRequest{}, err
		}

		policy := accessOnlyPolicy(access)
		request.NetworkPolicy = &policy
	}

	return request, nil
}

// templateName derives the name of the template sandbox whose snapshot a
// template's sandboxes start from. It covers the template identity and the
// environment selection.
func templateName(identity string, base createSandboxRequest) (string, error) {
	var selection any

	switch {
	case base.Source != nil:
		selection = base.Source
	case base.Image != "":
		selection = map[string]string{
			"image": base.Image,
		}
	default:
		selection = map[string]string{
			"runtime": base.Runtime,
		}
	}

	material, err := json.Marshal([]any{
		2,
		identity,
		selection,
	})
	if err != nil {
		return "", fmt.Errorf("encode vercel template name: %w", err)
	}

	digest := sha256.Sum256(material)

	return "ai-sdk-harness-v2-" + hex.EncodeToString(digest[:12]), nil
}
