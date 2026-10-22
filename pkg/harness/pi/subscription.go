package pi

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/credential"
)

// CredentialStore is application-owned storage for Pi's credential
// document: the JSON object Pi keeps in auth.json, keyed by provider ID,
// whose entries hold API keys and OAuth credentials such as Pi subscription
// logins.
//
// The adapter loads the document when a session starts and writes it into
// the session's private agent directory inside the sandbox, where Pi reads
// it and refreshes expiring OAuth tokens. When the session ends (Stop,
// Detach, SuspendTurn or Destroy) the adapter reads the document back and
// saves it when Pi changed it, so that rotated tokens are kept.
type CredentialStore interface {
	// Load returns the document, or nil when none is stored.
	Load(ctx context.Context) (json.RawMessage, error)

	// Save replaces the document.
	Save(ctx context.Context, document json.RawMessage) error
}

// FileCredentialStore stores the credential document in a file, such as
// Pi's own ~/.pi/agent/auth.json.
type FileCredentialStore struct {
	// Path is the absolute path of the file.
	Path string
}

// Load reads the file. A missing file stores no document.
func (s FileCredentialStore) Load(context.Context) (json.RawMessage, error) {
	content, err := os.ReadFile(s.Path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}

	if err != nil {
		return nil, fmt.Errorf("read pi credentials %s: %w", s.Path, err)
	}

	return content, nil
}

// Save replaces the file atomically, readable only by its owner. Saves of
// the same file are serialized within the process.
func (s FileCredentialStore) Save(_ context.Context, document json.RawMessage) error {
	release := credential.LockRefresh(s.Path)
	defer release()

	return credential.WriteFileAtomically(s.Path, document)
}

// credentialStoreFor returns the store a session uses: the configured
// store, or Pi's native auth.json when native subscription credentials
// apply. It returns nil when the session uses no stored credentials.
func credentialStoreFor(settings Settings, setup providerSetup, processEnv map[string]string) CredentialStore {
	if settings.Credentials != nil {
		return settings.Credentials
	}

	directory := nativeAgentDirectory(settings, setup, processEnv)
	if directory == "" {
		return nil
	}

	return FileCredentialStore{
		Path: filepath.Join(directory, "auth.json"),
	}
}

// nativeAgentDirectory ports upstream resolvePiSubscriptionAgentDir: Pi's
// native agent directory, whose auth.json holds subscription logins, unless
// the authentication is an explicit environment, the AI Gateway mode, or
// resolved AI Gateway credentials.
func nativeAgentDirectory(settings Settings, setup providerSetup, processEnv map[string]string) string {
	auth := settings.Auth
	if auth.Environment != nil || auth.Mode == harness.AuthenticationAIGateway || setup.usesGateway {
		return ""
	}

	switch {
	case settings.AgentDir != "":
		return absolutePath(settings.AgentDir)
	case processEnv["PI_CODING_AGENT_DIR"] != "":
		return absolutePath(processEnv["PI_CODING_AGENT_DIR"])
	}

	home := credential.HomeDirectory()
	if home == "" {
		return ""
	}

	return filepath.Join(home, ".pi", "agent")
}

// absolutePath resolves a host path against the working directory.
func absolutePath(directory string) string {
	resolved, err := filepath.Abs(directory)
	if err != nil {
		return directory
	}

	return resolved
}

// storedCredentials tracks the credential document of one session.
type storedCredentials struct {
	store CredentialStore

	// loaded is the document written into the sandbox, nil when the store
	// held none.
	loaded json.RawMessage
}

// loadCredentials loads the store's document.
func loadCredentials(ctx context.Context, store CredentialStore) (*storedCredentials, error) {
	if store == nil {
		return nil, nil
	}

	document, err := store.Load(ctx)
	if err != nil {
		return nil, err
	}

	if len(document) > 0 && !json.Valid(document) {
		return nil, errors.New("pi credentials are not a JSON document")
	}

	return &storedCredentials{
		store:  store,
		loaded: document,
	}, nil
}

// saveIfChanged saves a document Pi changed.
func (c *storedCredentials) saveIfChanged(ctx context.Context, document []byte) error {
	if c == nil || len(document) == 0 || bytes.Equal(document, c.loaded) || !json.Valid(document) {
		return nil
	}

	if err := c.store.Save(ctx, document); err != nil {
		return fmt.Errorf("save pi credentials: %w", err)
	}

	c.loaded = document

	return nil
}
