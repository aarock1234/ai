package vercel

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"sync"
	"time"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/sandbox"
)

// sessionPollInterval is the polling interval while a session stops.
const sessionPollInterval = 500 * time.Millisecond

var _ harness.NetworkSession = (*Session)(nil)

// Session is a Vercel Sandbox session. It implements harness.NetworkSession
// with ports, network policy, request transformations and port updates.
// Stop and Destroy act on the sandbox.
type Session struct {
	*core

	defaultWorkingDirectory string
	policy                  *policyManager
}

// core implements sandbox.Session for one sandbox. Operations on a stopped
// sandbox resume it and retry once, as the upstream SDK does.
type core struct {
	client *apiClient
	name   string

	mu            sync.Mutex
	sessionID     string
	cwd           string
	routes        []route
	sessionPolicy *apiNetworkPolicy

	resumeMu sync.Mutex
}

// restricted is the view returned by Session.Restricted.
type restricted struct {
	*core
}

// newSession wraps a sandbox and its current session.
func newSession(client *apiClient, state sandboxAndSession) *Session {
	shared := &core{
		client: client,
		name:   state.Sandbox.Name,
	}
	shared.adopt(state)

	session := &Session{
		core:                    shared,
		defaultWorkingDirectory: state.Session.Cwd,
	}

	session.policy = &policyManager{
		current: shared.currentPolicy,
		apply:   shared.applyPolicy,
	}

	return session
}

// ID returns the sandbox name. Persist it to resume the sandbox with Resume.
func (s *Session) ID() string {
	return s.name
}

// DefaultWorkingDirectory returns the working directory of the sandbox's
// session when it was created.
func (s *Session) DefaultWorkingDirectory() string {
	return s.defaultWorkingDirectory
}

// Ports returns the exposed ports.
func (s *Session) Ports() []int {
	s.mu.Lock()
	defer s.mu.Unlock()

	ports := make([]int, 0, len(s.routes))
	for _, exposed := range s.routes {
		ports = append(ports, exposed.Port)
	}

	return ports
}

// PortEndpoint returns the public URL of an exposed port. The WebSocket
// protocol selects ws or wss after the public URL's scheme.
func (s *Session) PortEndpoint(_ context.Context, options harness.PortEndpointOptions) (harness.PortEndpoint, error) {
	s.mu.Lock()
	index := slices.IndexFunc(s.routes, func(exposed route) bool {
		return exposed.Port == options.Port
	})

	var exposed route
	if index >= 0 {
		exposed = s.routes[index]
	}
	s.mu.Unlock()

	if index < 0 {
		return harness.PortEndpoint{}, &harness.CapabilityUnsupportedError{
			Message:   fmt.Sprintf("Port %d is not exposed on this sandbox. Exposed ports: %v.", options.Port, s.Ports()),
			HarnessID: providerID,
		}
	}

	endpoint, err := url.Parse("https://" + exposed.Subdomain + ".vercel.run")
	if err != nil {
		return harness.PortEndpoint{}, fmt.Errorf("parse vercel sandbox port url: %w", err)
	}

	secure := endpoint.Scheme == "https"

	switch options.Protocol {
	case harness.PortHTTP:
		endpoint.Scheme = schemeFor(secure, "https", "http")
	case harness.PortWebSocket:
		endpoint.Scheme = schemeFor(secure, "wss", "ws")
	default:
		endpoint.Scheme = "https"
	}

	if endpoint.Path == "" {
		endpoint.Path = "/"
	}

	return harness.PortEndpoint{
		URL: endpoint.String(),
	}, nil
}

// SetPorts replaces the exposed ports.
func (s *Session) SetPorts(ctx context.Context, ports []int) error {
	if ports == nil {
		ports = []int{}
	}

	response, err := s.client.updateSandbox(ctx, s.name, updateSandboxRequest{
		Ports: ports,
	})
	if err != nil {
		return fmt.Errorf("set vercel sandbox ports: %w", err)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if response.Routes != nil {
		s.routes = response.Routes
	}

	return nil
}

// SetNetworkPolicy replaces the outbound network access, keeping the
// request transformations the session manages.
func (s *Session) SetNetworkPolicy(ctx context.Context, policy harness.NetworkPolicy) error {
	return s.policy.setNetworkPolicy(ctx, policy)
}

// SetRequestTransformations replaces every request transformation.
func (s *Session) SetRequestTransformations(ctx context.Context, transformations []harness.RequestTransformation) error {
	return s.policy.setRequestTransformations(ctx, transformations)
}

// AddRequestTransformations adds request transformations without replacing
// unrelated ones.
func (s *Session) AddRequestTransformations(ctx context.Context, transformations []harness.RequestTransformation) error {
	return s.policy.addRequestTransformations(ctx, transformations)
}

// Stop stops the sandbox's session.
func (s *Session) Stop(ctx context.Context) error {
	err := s.client.stopSession(ctx, s.currentSessionID())
	if err != nil && !isAPIStatus(err, http.StatusGone, "") {
		return fmt.Errorf("stop vercel sandbox %s: %w", s.name, err)
	}

	return nil
}

// Destroy stops the sandbox and deletes it.
func (s *Session) Destroy(ctx context.Context) error {
	_ = s.Stop(ctx)

	if err := s.client.deleteSandbox(ctx, s.name); err != nil && !isAPIStatus(err, http.StatusNotFound, "") {
		return fmt.Errorf("delete vercel sandbox %s: %w", s.name, err)
	}

	return nil
}

// Restricted returns the files-and-processes view of the sandbox.
func (s *Session) Restricted() sandbox.Session {
	return &restricted{
		core: s.core,
	}
}

// Description describes the sandbox for the agent's instructions.
func (c *core) Description() string {
	return fmt.Sprintf("Vercel Sandbox (name: %s).\nFilesystem changes persist for the lifetime of the sandbox.", c.name)
}

// adopt records the state of the sandbox's current session.
func (c *core) adopt(state sandboxAndSession) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.sessionID = state.Session.ID
	c.cwd = state.Session.Cwd
	c.routes = state.Routes
	c.sessionPolicy = state.Session.NetworkPolicy
}

// currentSessionID returns the ID of the current session.
func (c *core) currentSessionID() string {
	c.mu.Lock()
	defer c.mu.Unlock()

	return c.sessionID
}

// workingDirectory returns the current session's working directory.
func (c *core) workingDirectory() string {
	c.mu.Lock()
	defer c.mu.Unlock()

	return c.cwd
}

// currentPolicy returns the live network policy in the request shape.
func (c *core) currentPolicy() vercelPolicy {
	c.mu.Lock()
	defer c.mu.Unlock()

	return fromAPINetworkPolicy(c.sessionPolicy)
}

// applyPolicy stores a policy on the sandbox and applies it to the running
// session.
func (c *core) applyPolicy(ctx context.Context, policy vercelPolicy) error {
	if _, err := c.client.updateSandbox(ctx, c.name, updateSandboxRequest{
		NetworkPolicy: &policy,
	}); err != nil {
		return fmt.Errorf("update vercel sandbox network policy: %w", err)
	}

	session, err := c.client.updateNetworkPolicy(ctx, c.currentSessionID(), policy)
	if isStoppedError(err) {
		return nil
	}

	if err != nil {
		return fmt.Errorf("apply vercel sandbox network policy: %w", err)
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	c.sessionPolicy = session.NetworkPolicy

	return nil
}

// withResume runs operation against the current session and, when the
// session stopped, resumes the sandbox and runs it once more.
func (c *core) withResume(ctx context.Context, operation func(sessionID string) error) error {
	err := operation(c.currentSessionID())

	switch {
	case isAPIStatus(err, http.StatusGone, ""):
		if err := c.resume(ctx); err != nil {
			return err
		}
	case isStoppingError(err):
		if err := c.waitForStop(ctx); err != nil {
			return err
		}

		if err := c.resume(ctx); err != nil {
			return err
		}
	default:
		return err
	}

	return operation(c.currentSessionID())
}

// resume starts a new session of the sandbox.
func (c *core) resume(ctx context.Context) error {
	c.resumeMu.Lock()
	defer c.resumeMu.Unlock()

	state, err := c.client.getSandbox(ctx, c.name, true)
	if err != nil {
		return fmt.Errorf("resume vercel sandbox %s: %w", c.name, err)
	}

	c.adopt(state)

	return nil
}

// waitForStop polls the session until it is no longer stopping or
// snapshotting.
func (c *core) waitForStop(ctx context.Context) error {
	for {
		session, err := c.client.getSession(ctx, c.currentSessionID())
		if err != nil {
			return fmt.Errorf("poll vercel sandbox session: %w", err)
		}

		if session.Status != "stopping" && session.Status != "snapshotting" {
			return nil
		}

		timer := time.NewTimer(sessionPollInterval)

		select {
		case <-ctx.Done():
			timer.Stop()

			return context.Cause(ctx)
		case <-timer.C:
		}
	}
}

// isStoppingError reports an operation on a stopping or snapshotting
// session.
func isStoppingError(err error) bool {
	return isAPIStatus(err, http.StatusUnprocessableEntity, "sandbox_stopping") ||
		isAPIStatus(err, http.StatusUnprocessableEntity, "sandbox_snapshotting")
}

// isStoppedError reports an operation on a stopped or stopping session.
func isStoppedError(err error) bool {
	return isAPIStatus(err, http.StatusGone, "") || isAPIStatus(err, http.StatusUnprocessableEntity, "sandbox_stopping")
}

// schemeFor returns the secure or plain scheme.
func schemeFor(secure bool, secureScheme string, plainScheme string) string {
	if secure {
		return secureScheme
	}

	return plainScheme
}

// errNoCommand reports a command stream that ended before the command
// finished.
var errNoCommand = errors.New("vercel sandbox command stream ended before the command finished")
