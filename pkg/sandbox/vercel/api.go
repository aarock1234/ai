package vercel

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
)

// sandboxMetadata is the part of a sandbox the session reads.
type sandboxMetadata struct {
	Name             string            `json:"name"`
	Status           string            `json:"status"`
	CurrentSessionID string            `json:"currentSessionId"`
	Cwd              string            `json:"cwd"`
	NetworkPolicy    *apiNetworkPolicy `json:"networkPolicy"`
}

// sessionMetadata is the part of a sandbox session (the running VM) the
// session reads.
type sessionMetadata struct {
	ID            string            `json:"id"`
	Status        string            `json:"status"`
	Cwd           string            `json:"cwd"`
	NetworkPolicy *apiNetworkPolicy `json:"networkPolicy"`
}

// route maps an exposed port to its subdomain.
type route struct {
	URL       string `json:"url"`
	Subdomain string `json:"subdomain"`
	Port      int    `json:"port"`
}

// sandboxAndSession is the response of sandbox creation and lookup.
type sandboxAndSession struct {
	Sandbox sandboxMetadata `json:"sandbox"`
	Session sessionMetadata `json:"session"`
	Routes  []route         `json:"routes"`
	Resumed bool            `json:"resumed"`
}

// commandMetadata describes a command.
type commandMetadata struct {
	ID       string `json:"id"`
	ExitCode *int   `json:"exitCode"`
}

// logLine is one line of a command log stream.
type logLine struct {
	Stream string          `json:"stream"`
	Data   json.RawMessage `json:"data"`
}

// snapshotMetadata describes a snapshot.
type snapshotMetadata struct {
	ID             string `json:"id"`
	Status         string `json:"status"`
	CreationMethod string `json:"creationMethod"`
}

// createSandboxRequest is the body of sandbox creation.
type createSandboxRequest struct {
	ProjectID          string            `json:"projectId"`
	Name               string            `json:"name,omitempty"`
	Ports              []int             `json:"ports"`
	Source             *Source           `json:"source,omitempty"`
	Timeout            int64             `json:"timeout,omitempty"`
	Resources          *resources        `json:"resources,omitempty"`
	Runtime            string            `json:"runtime,omitempty"`
	Image              string            `json:"image,omitempty"`
	Persistent         *bool             `json:"persistent,omitempty"`
	NetworkPolicy      *vercelPolicy     `json:"networkPolicy,omitempty"`
	Env                map[string]string `json:"env,omitempty"`
	Tags               map[string]string `json:"tags,omitempty"`
	SnapshotExpiration *int64            `json:"snapshotExpiration,omitempty"`
	Region             string            `json:"region,omitempty"`
}

// resources are the sandbox's compute resources.
type resources struct {
	VCPUs int `json:"vcpus"`
}

// createSandbox creates a sandbox. Requests with a runtime use the v2
// endpoint and image-based requests the v3 endpoint, as upstream does.
func (c *apiClient) createSandbox(ctx context.Context, body createSandboxRequest) (sandboxAndSession, error) {
	endpoint := "/v3/sandboxes"
	if body.Runtime != "" {
		endpoint = "/v2/sandboxes"
	}

	var result sandboxAndSession
	err := c.doJSON(ctx, apiRequest{
		method: http.MethodPost,
		path:   endpoint,
		body:   body,
	}, &result)

	return result, err
}

// getSandbox looks a sandbox up by name, resuming its session when resume
// is set.
func (c *apiClient) getSandbox(ctx context.Context, name string, resume bool) (sandboxAndSession, error) {
	query := url.Values{
		"projectId": {
			c.credentials.ProjectID,
		},
	}

	if resume {
		query.Set("resume", "true")
	}

	var result sandboxAndSession
	err := c.doJSON(ctx, apiRequest{
		method: http.MethodGet,
		path:   "/v2/sandboxes/" + url.PathEscape(name),
		query:  query,
	}, &result)

	return result, err
}

// updateSandboxRequest is the body of a sandbox update.
type updateSandboxRequest struct {
	Ports         []int         `json:"ports,omitempty"`
	NetworkPolicy *vercelPolicy `json:"networkPolicy,omitempty"`
}

// updateSandboxResponse is the response of a sandbox update.
type updateSandboxResponse struct {
	Sandbox sandboxMetadata `json:"sandbox"`
	Routes  []route         `json:"routes"`
}

// updateSandbox updates the sandbox's configuration.
func (c *apiClient) updateSandbox(ctx context.Context, name string, body updateSandboxRequest) (updateSandboxResponse, error) {
	var result updateSandboxResponse
	err := c.doJSON(ctx, apiRequest{
		method: http.MethodPatch,
		path:   "/v2/sandboxes/" + url.PathEscape(name),
		query:  c.projectQuery(),
		body:   body,
	}, &result)

	return result, err
}

// deleteSandbox deletes a sandbox.
func (c *apiClient) deleteSandbox(ctx context.Context, name string) error {
	return c.doJSON(ctx, apiRequest{
		method: http.MethodDelete,
		path:   "/v2/sandboxes/" + url.PathEscape(name),
		query:  c.projectQuery(),
	}, nil)
}

// getSession returns a session's state.
func (c *apiClient) getSession(ctx context.Context, sessionID string) (sessionMetadata, error) {
	var result struct {
		Session sessionMetadata `json:"session"`
	}

	err := c.doJSON(ctx, apiRequest{
		method: http.MethodGet,
		path:   sessionPath(sessionID, ""),
	}, &result)

	return result.Session, err
}

// stopSession stops a session.
func (c *apiClient) stopSession(ctx context.Context, sessionID string) error {
	return c.doJSON(ctx, apiRequest{
		method: http.MethodPost,
		path:   sessionPath(sessionID, "/stop"),
	}, nil)
}

// updateNetworkPolicy applies a network policy to the running session.
func (c *apiClient) updateNetworkPolicy(ctx context.Context, sessionID string, policy vercelPolicy) (sessionMetadata, error) {
	var result struct {
		Session sessionMetadata `json:"session"`
	}

	err := c.doJSON(ctx, apiRequest{
		method: http.MethodPost,
		path:   sessionPath(sessionID, "/network-policy"),
		body:   policy,
	}, &result)

	return result.Session, err
}

// createSnapshot snapshots a session, which stops it.
func (c *apiClient) createSnapshot(ctx context.Context, sessionID string, expiration int64) (snapshotMetadata, error) {
	var result struct {
		Snapshot snapshotMetadata `json:"snapshot"`
	}

	err := c.doJSON(ctx, apiRequest{
		method: http.MethodPost,
		path:   sessionPath(sessionID, "/snapshot"),
		body: map[string]int64{
			"expiration": expiration,
		},
	}, &result)

	return result.Snapshot, err
}

// listSnapshots lists the newest snapshots of a sandbox name.
func (c *apiClient) listSnapshots(ctx context.Context, name string) ([]snapshotMetadata, error) {
	var result struct {
		Snapshots []snapshotMetadata `json:"snapshots"`
	}

	err := c.doJSON(ctx, apiRequest{
		method: http.MethodGet,
		path:   "/v2/sandboxes/snapshots",
		query: url.Values{
			"project": {
				c.credentials.ProjectID,
			},
			"name": {
				name,
			},
			"sortOrder": {
				"desc",
			},
		},
	}, &result)

	return result.Snapshots, err
}

// commandRequest is the body of a command run.
type commandRequest struct {
	Command string            `json:"command"`
	Args    []string          `json:"args"`
	Cwd     string            `json:"cwd,omitempty"`
	Env     map[string]string `json:"env"`
	Sudo    bool              `json:"sudo"`
	Wait    bool              `json:"wait,omitempty"`
	Logs    bool              `json:"logs,omitempty"`
}

// startCommand starts a detached command.
func (c *apiClient) startCommand(ctx context.Context, sessionID string, body commandRequest) (commandMetadata, error) {
	var result struct {
		Command commandMetadata `json:"command"`
	}

	err := c.doJSON(ctx, apiRequest{
		method: http.MethodPost,
		path:   sessionPath(sessionID, "/cmd"),
		body:   body,
	}, &result)

	return result.Command, err
}

// runCommandStream starts a command that waits for its exit and returns the
// newline-delimited JSON stream of its logs. The caller closes the body.
func (c *apiClient) runCommandStream(ctx context.Context, sessionID string, body commandRequest) (io.ReadCloser, error) {
	body.Wait = true
	body.Logs = true

	response, err := c.do(ctx, apiRequest{
		method:  http.MethodPost,
		path:    sessionPath(sessionID, "/cmd"),
		body:    body,
		noRetry: true,
	})
	if err != nil {
		return nil, err
	}

	return response.Body, nil
}

// waitCommand waits for a command to finish.
func (c *apiClient) waitCommand(ctx context.Context, sessionID string, commandID string) (commandMetadata, error) {
	var result struct {
		Command commandMetadata `json:"command"`
	}

	err := c.doJSON(ctx, apiRequest{
		method: http.MethodGet,
		path:   sessionPath(sessionID, "/cmd/"+url.PathEscape(commandID)),
		query: url.Values{
			"wait": {
				"true",
			},
		},
	}, &result)

	return result.Command, err
}

// commandLogs returns the log stream of a command. The caller closes the
// body.
func (c *apiClient) commandLogs(ctx context.Context, sessionID string, commandID string) (io.ReadCloser, error) {
	response, err := c.do(ctx, apiRequest{
		method: http.MethodGet,
		path:   sessionPath(sessionID, "/cmd/"+url.PathEscape(commandID)+"/logs"),
	})
	if err != nil {
		return nil, err
	}

	return response.Body, nil
}

// killCommand sends SIGTERM to a command.
func (c *apiClient) killCommand(ctx context.Context, sessionID string, commandID string) error {
	return c.doJSON(ctx, apiRequest{
		method: http.MethodPost,
		path:   sessionPath(sessionID, "/cmd/"+url.PathEscape(commandID)+"/kill"),
		body: map[string]int{
			"signal": 15,
		},
	}, nil)
}

// readFile returns the content of a file, or false when it does not exist.
// The caller closes the body.
func (c *apiClient) readFile(ctx context.Context, sessionID string, path string, cwd string) (io.ReadCloser, bool, error) {
	response, err := c.do(ctx, apiRequest{
		method: http.MethodPost,
		path:   sessionPath(sessionID, "/fs/read"),
		body: map[string]string{
			"path": path,
			"cwd":  cwd,
		},
	})
	if isAPIStatus(err, http.StatusNotFound, "") {
		return nil, false, nil
	}

	if err != nil {
		return nil, false, err
	}

	return response.Body, true, nil
}

// writeFiles extracts a gzipped tar archive at extractDir.
func (c *apiClient) writeFiles(ctx context.Context, sessionID string, archive []byte, extractDir string) error {
	return c.doJSON(ctx, apiRequest{
		method:      http.MethodPost,
		path:        sessionPath(sessionID, "/fs/write"),
		rawBody:     archive,
		contentType: "application/gzip",
		header: http.Header{
			"X-Cwd": {
				extractDir,
			},
		},
	}, nil)
}

// projectQuery scopes a request to the project.
func (c *apiClient) projectQuery() url.Values {
	return url.Values{
		"projectId": {
			c.credentials.ProjectID,
		},
	}
}

// sessionPath returns the path of a session endpoint.
func sessionPath(sessionID string, suffix string) string {
	return "/v2/sandboxes/sessions/" + url.PathEscape(sessionID) + suffix
}

// readLogLines decodes a newline-delimited JSON log stream, calling handle
// for each log line and onCommand, when set, for a command that started,
// until the stream ends or a finished command arrives. It returns the
// finished command when the stream reports one.
func readLogLines(stream io.Reader, handle func(line logLine), onCommand func(command commandMetadata)) (*commandMetadata, error) {
	scanner := bufio.NewScanner(stream)
	scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)

	for scanner.Scan() {
		raw := scanner.Bytes()
		if len(raw) == 0 {
			continue
		}

		var line struct {
			logLine
			Command *commandMetadata `json:"command"`
		}

		if err := json.Unmarshal(raw, &line); err != nil {
			return nil, fmt.Errorf("decode vercel sandbox log line: %w", err)
		}

		if line.Command != nil {
			if line.Command.ExitCode != nil {
				return line.Command, nil
			}

			if onCommand != nil {
				onCommand(*line.Command)
			}

			continue
		}

		if line.Stream == "error" {
			return nil, logStreamError(line.Data)
		}

		handle(line.logLine)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read vercel sandbox log stream: %w", err)
	}

	return nil, nil
}

// logText returns the text of a stdout or stderr log line.
func logText(line logLine) string {
	var text string
	if err := json.Unmarshal(line.Data, &text); err != nil {
		return string(line.Data)
	}

	return text
}

// logStreamError converts an error line of a log stream.
func logStreamError(data json.RawMessage) error {
	var failure struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	}

	if err := json.Unmarshal(data, &failure); err != nil {
		return errors.New("vercel sandbox log stream failed")
	}

	return fmt.Errorf("vercel sandbox log stream failed (%s): %s", failure.Code, failure.Message)
}
