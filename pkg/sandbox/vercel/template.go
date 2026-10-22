package vercel

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"time"

	"github.com/aarock1234/ai/pkg/harness"
)

// Template snapshot polling, as upstream.
const (
	snapshotPollInterval = 500 * time.Millisecond
	snapshotWaitTimeout  = 60 * time.Second
)

// nameConflictPattern matches the API's message for a taken sandbox name.
var nameConflictPattern = regexp.MustCompile(`(?i)sandbox.*name.*already exists`)

// templateCreation describes a sandbox created from a template.
type templateCreation struct {
	client    *apiClient
	base      createSandboxRequest
	template  harness.SandboxTemplate
	sandboxID string
}

// createFromTemplate ensures the template's snapshot and starts a sandbox
// from it.
func createFromTemplate(ctx context.Context, creation templateCreation) (*Session, error) {
	name, err := templateName(creation.template.Identity, creation.base)
	if err != nil {
		return nil, err
	}

	snapshotID, err := ensureTemplateSnapshot(ctx, creation, name)
	if err != nil {
		return nil, fmt.Errorf("prepare vercel sandbox template %s: %w", name, err)
	}

	live := creation.base
	live.Runtime = ""
	live.Image = ""
	live.Persistent = nil
	live.Name = creation.sandboxID
	live.Source = &Source{
		Type:       SourceSnapshot,
		SnapshotID: snapshotID,
	}

	state, err := creation.client.createSandbox(ctx, live)
	if err != nil {
		return nil, fmt.Errorf("create vercel sandbox from template: %w", err)
	}

	return newSession(creation.client, state), nil
}

// ensureTemplateSnapshot returns the snapshot of the template sandbox. A
// missing template sandbox is created, prepared and snapshotted; an existing
// one, or one another process is creating, publishes its snapshot, which is
// polled for.
func ensureTemplateSnapshot(ctx context.Context, creation templateCreation, name string) (string, error) {
	_, err := creation.client.getSandbox(ctx, name, false)

	switch {
	case err == nil:
		return pollTemplateSnapshot(ctx, creation.client, name)
	case isAPIStatus(err, http.StatusNotFound, ""):
	case isAPIStatus(err, http.StatusGone, "snapshot_not_found"):
		if err := creation.client.deleteSandbox(ctx, name); err != nil && !isAPIStatus(err, http.StatusNotFound, "") {
			return "", err
		}
	default:
		return "", err
	}

	snapshotID, started, err := prepareTemplateSnapshot(ctx, creation, name)
	if err == nil {
		return snapshotID, nil
	}

	if started || !isNameConflict(err) {
		return "", err
	}

	return pollTemplateSnapshot(ctx, creation.client, name)
}

// prepareTemplateSnapshot creates the template sandbox, applies the
// template and snapshots it. It reports whether preparation started.
func prepareTemplateSnapshot(ctx context.Context, creation templateCreation, name string) (string, bool, error) {
	var expiration int64
	if creation.base.SnapshotExpiration != nil {
		expiration = *creation.base.SnapshotExpiration
	}

	persistent := false
	request := creation.base
	request.Name = name
	request.Persistent = &persistent
	request.SnapshotExpiration = &expiration

	state, err := creation.client.createSandbox(ctx, request)
	if err != nil {
		return "", false, err
	}

	session := newSession(creation.client, state)

	if err := creation.template.Prepare(ctx, session.Restricted()); err != nil {
		return "", true, errors.Join(err, session.Stop(context.WithoutCancel(ctx)))
	}

	snapshot, err := creation.client.createSnapshot(ctx, session.currentSessionID(), expiration)
	if err != nil {
		return "", true, fmt.Errorf("snapshot vercel template sandbox: %w", err)
	}

	return snapshot.ID, true, nil
}

// pollTemplateSnapshot waits for the template's manual snapshot to publish.
func pollTemplateSnapshot(ctx context.Context, client *apiClient, name string) (string, error) {
	deadline := time.Now().Add(snapshotWaitTimeout)

	for time.Now().Before(deadline) {
		snapshots, err := client.listSnapshots(ctx, name)
		if err != nil {
			return "", err
		}

		for _, snapshot := range snapshots {
			if snapshot.Status == "created" && snapshot.CreationMethod == "manual" {
				return snapshot.ID, nil
			}
		}

		timer := time.NewTimer(snapshotPollInterval)

		select {
		case <-ctx.Done():
			timer.Stop()

			return "", context.Cause(ctx)
		case <-timer.C:
		}
	}

	return "", fmt.Errorf("timed out waiting for snapshot of template %q to publish", name)
}

// isNameConflict reports a creation that failed because the name is taken.
func isNameConflict(err error) bool {
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		return false
	}

	isConflictStatus := apiErr.StatusCode == http.StatusBadRequest || apiErr.StatusCode == http.StatusConflict

	return isConflictStatus && nameConflictPattern.MatchString(apiErr.Message)
}
