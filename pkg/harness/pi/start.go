package pi

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"

	"github.com/aarock1234/ai/internal/shellquote"
	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
	"github.com/aarock1234/ai/pkg/sandbox"
)

// Start starts a Pi session: it resolves and brokers the credentials, then
// reattaches to the Pi process an earlier Detach or SuspendTurn left
// running, or launches Pi, opening the persisted session file of a resumed
// session.
func (h *Harness) Start(ctx context.Context, options harness.StartOptions) (harness.Session, error) {
	data, err := bridgeadapter.DecodeStateData[stateData](options, HarnessID)
	if err != nil {
		return nil, err
	}

	if err := data.validate(); err != nil {
		return nil, err
	}

	restricted := harness.RestrictedSession(options.SandboxSession)

	homeDir, err := harness.ResolveSandboxHomeDir(ctx, restricted)
	if err != nil {
		return nil, err
	}

	sessionLayout, err := newLayout(homeDir, options.SessionWorkDir, options.SessionID)
	if err != nil {
		return nil, err
	}

	processEnv := bridgeadapter.HostEnvironment()
	setup := resolveProviders(h.settings.Auth, processEnv, bridgeadapter.HeaderMap(options.Headers))

	credentials, err := h.sandboxCredentials(ctx, sandboxCredentialsOptions{
		start:    options,
		setup:    setup,
		previous: data.SandboxCredentialEnvironment,
	})
	if err != nil {
		return nil, err
	}

	stored, err := loadCredentials(ctx, credentialStoreFor(h.settings, setup, processEnv))
	if err != nil {
		return nil, err
	}

	launch, err := h.launchConfiguration(launchInput{
		start:      options,
		layout:     sessionLayout,
		setup:      setup,
		sandboxEnv: credentials.sandboxEnv,
	})
	if err != nil {
		return nil, err
	}

	current := &session{
		harness:               h,
		sessionID:             options.SessionID,
		isResume:              options.ResumeFrom != nil || options.ContinueFrom != nil,
		sandbox:               restricted,
		layout:                sessionLayout,
		fingerprint:           launch.fingerprint,
		setup:                 setup,
		credentials:           stored,
		credentialEnvironment: credentials.credentialEnv,
		builtinToolNames:      activeBuiltinToolNames(options.BuiltinToolFiltering),
		observability:         options.Observability,
		resultDialogs:         make(map[string]string),
		approvalDialogs:       make(map[string]string),
		earlyResults:          make(map[string]hostToolResult),
		earlyApprovals:        make(map[string]approvalDecision),
		sessionFileName:       data.SessionFileName,
	}

	current.translator = newTranslator(current.builtinToolNames, nil)

	attached, err := h.reattach(ctx, current, data)
	if err != nil {
		return nil, err
	}

	if attached {
		return current, nil
	}

	if err := current.launch(ctx, launch); err != nil {
		return nil, err
	}

	current.rerunContinue = options.ContinueFrom != nil

	return current, nil
}

// reattach attaches the session to the Pi process the lifecycle state
// locates, when it still runs with the same configuration. A process with
// another configuration is stopped, so that Pi restarts from its session
// file with the current settings.
func (h *Harness) reattach(ctx context.Context, current *session, data stateData) (bool, error) {
	runtime := data.Runtime
	if runtime == nil {
		return false, nil
	}

	stateDir := current.layout.relayStateDir(runtime.LaunchID)

	status, err := relayState(ctx, current.sandbox, stateDir)
	if err != nil {
		return false, err
	}

	if status != relayRunning {
		return false, nil
	}

	if !h.reattachLiveRuntime() || runtime.ConfigurationFingerprint != current.fingerprint {
		signalRelay(ctx, current.sandbox, stateDir)

		return false, nil
	}

	current.launchID = runtime.LaunchID
	current.conn = newConnection(connectionOptions{
		sandbox:       current.sandbox,
		layout:        current.layout,
		launchID:      runtime.LaunchID,
		inputSequence: runtime.InputSequence,
		outboxOffset:  runtime.OutboxOffset,
		handler:       current.handleRecord,
	})

	current.restore(*runtime)
	current.diagnostics = followDiagnostics(ctx, current, current.observability)

	return true, nil
}

// restore restores the host state of a live runtime.
func (s *session) restore(runtime liveRuntime) {
	if runtime.Translator != nil {
		s.translator = runtime.Translator
	}

	for _, dialog := range runtime.PendingDialogs {
		switch dialog.Kind {
		case dialogToolResult:
			s.resultDialogs[dialog.ToolCallID] = dialog.RequestID
		case dialogToolApproval:
			s.approvalDialogs[dialog.ToolCallID] = dialog.RequestID
		}
	}

	s.turnBaseline = runtime.TurnBaseline
	s.toolsSignature = runtime.ToolsSignature
	s.modelSelected = runtime.ModelSelected
	s.liveTurn = runtime.TurnActive

	s.model = runtime.Model
}

// launch writes the session's files, starts the relay and waits until Pi
// answers.
func (s *session) launch(ctx context.Context, launch launchConfiguration) error {
	if err := s.writeLaunchFiles(ctx, launch); err != nil {
		return err
	}

	args, err := s.launchArgs(ctx, launch)
	if err != nil {
		return err
	}

	s.launchID = newLaunchID()
	s.conn = newConnection(connectionOptions{
		sandbox:  s.sandbox,
		layout:   s.layout,
		launchID: s.launchID,
		handler:  s.handleRecord,
	})

	relay, err := spawnRelay(ctx, relayOptions{
		sandbox:  s.sandbox,
		layout:   s.layout,
		launchID: s.launchID,
		args:     args,
		env:      launch.env,
	})
	if err != nil {
		return err
	}

	s.relay = relay
	s.diagnostics = followDiagnostics(ctx, s, s.observability)

	if err := s.awaitReady(ctx); err != nil {
		tail := stderrTail(ctx, s.sandbox, s.conn.stateDir())
		signalRelay(context.WithoutCancel(ctx), s.sandbox, s.conn.stateDir())
		s.conn.stop()
		s.diagnostics.stop()
		_ = relay.Kill(context.WithoutCancel(ctx))

		return withStderr(fmt.Errorf("start pi: %w", err), tail)
	}

	return nil
}

// awaitReady waits until Pi answers its first commands, fails when the
// adapter's extension reported registration errors, and records Pi's
// session file and model.
func (s *session) awaitReady(ctx context.Context) error {
	readyCtx, cancel := context.WithTimeout(ctx, s.harness.startupTimeout())
	defer cancel()

	if err := s.conn.start(readyCtx); err != nil {
		return err
	}

	reply, err := s.conn.call(readyCtx, simpleCommand{
		Type: "get_commands",
	})
	if err != nil {
		return err
	}

	var commands commandsData
	if err := json.Unmarshal(reply.Data, &commands); err != nil {
		return fmt.Errorf("decode pi commands: %w", err)
	}

	for _, command := range commands.Commands {
		if command.Name == startupErrorCommand {
			return errors.New(command.Description)
		}
	}

	reply, err = s.conn.call(readyCtx, simpleCommand{
		Type: "get_state",
	})
	if err != nil {
		return err
	}

	var state stateResponseData
	if err := json.Unmarshal(reply.Data, &state); err != nil {
		return fmt.Errorf("decode pi state: %w", err)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if name := path.Base(state.SessionFile); state.SessionFile != "" && isSafeSessionFileName(name) {
		s.sessionFileName = name
	}

	s.model = state.Model

	return nil
}

// sandboxCredentialsOptions configure sandboxCredentials.
type sandboxCredentialsOptions struct {
	start    harness.StartOptions
	setup    providerSetup
	previous map[string]string
}

// sandboxCredentials are the credentials Pi receives.
type sandboxCredentials struct {
	sandboxEnv map[string]string

	// credentialEnv holds the placeholders of brokered credentials, and is
	// nil when the sandbox cannot broker them.
	credentialEnv map[string]string
}

// sandboxCredentials brokers the credentials when the sandbox can add
// request transformations, and forwards them otherwise.
func (h *Harness) sandboxCredentials(ctx context.Context, options sandboxCredentialsOptions) (sandboxCredentials, error) {
	brokering := bridgeadapter.BrokeringOptions{
		Sandbox:                        options.start.SandboxSession,
		Environment:                    options.setup.environment,
		CredentialEnvironmentVariables: options.setup.credentialVariables(),
		Forwarding:                     h.settings.CredentialForwarding,
		Previous:                       options.previous,
		Transformations:                options.setup.requestTransformations,
	}

	brokered, ok, err := bridgeadapter.BrokerCredentials(ctx, brokering)
	if err != nil {
		return sandboxCredentials{}, err
	}

	if ok {
		return sandboxCredentials{
			sandboxEnv:    brokered.SandboxEnvironment,
			credentialEnv: brokered.CredentialEnvironment,
		}, nil
	}

	forwarded, err := bridgeadapter.ForwardCredentials(ctx, brokering)
	if err != nil {
		return sandboxCredentials{}, err
	}

	return sandboxCredentials{
		sandboxEnv: forwarded,
	}, nil
}

// launchInput are the inputs of launchConfiguration.
type launchInput struct {
	start      harness.StartOptions
	layout     layout
	setup      providerSetup
	sandboxEnv map[string]string
}

// launchConfiguration is everything a launch writes and passes to Pi.
type launchConfiguration struct {
	config     []byte
	env        map[string]string
	extensions []Extension

	// agentFiles are files copied from the host agent directory, keyed by
	// file name.
	agentFiles map[string][]byte

	usesMCP     bool
	fingerprint string
}

// launchConfiguration builds the launch configuration and its fingerprint.
func (h *Harness) launchConfiguration(input launchInput) (launchConfiguration, error) {
	permissionMode := input.start.PermissionMode
	if permissionMode == "" {
		permissionMode = harness.PermissionAllowAll
	}

	providers := slices.Clone(input.setup.registrations)
	for _, name := range slices.Sorted(maps.Keys(h.settings.Providers)) {
		providers = append(providers, namedConfig{
			Name:   name,
			Config: h.settings.Providers[name],
		})
	}

	mcpServers := mcpServerRegistrations(h.settings.MCPServers)

	config, err := json.Marshal(extensionConfig{
		CWD:            input.layout.workDir,
		TurnFile:       input.layout.turnFile(),
		SkillsDir:      input.layout.skillsRoot(),
		BuiltinTools:   activeBuiltinToolNames(input.start.BuiltinToolFiltering),
		PermissionMode: string(permissionMode),
		PathPolicy:     h.settings.FileToolPathPolicy,
		Providers:      providers,
		MCPServers:     mcpServers,
	})
	if err != nil {
		return launchConfiguration{}, fmt.Errorf("encode pi extension configuration: %w", err)
	}

	agentFiles, err := h.readAgentFiles()
	if err != nil {
		return launchConfiguration{}, err
	}

	launch := launchConfiguration{
		config:     config,
		env:        h.launchEnvironment(input),
		extensions: h.settings.Extensions,
		agentFiles: agentFiles,
		usesMCP:    len(mcpServers) > 0,
	}

	fingerprint, err := launch.computeFingerprint(h.settings.ThinkingLevel)
	if err != nil {
		return launchConfiguration{}, err
	}

	launch.fingerprint = fingerprint

	return launch, nil
}

// launchEnvironment returns Pi's process environment: the brokered or
// forwarded credentials, Pi's isolated agent directory, the extension's
// files, and switches that keep Pi from automatic network activity, as
// upstream's in-process runtime does not refresh model catalogs.
func (h *Harness) launchEnvironment(input launchInput) map[string]string {
	env := maps.Clone(input.sandboxEnv)
	if env == nil {
		env = make(map[string]string)
	}

	env["PI_CODING_AGENT_DIR"] = input.layout.agentDir()
	env["AI_SDK_HARNESS_PI_CONFIG"] = input.layout.extensionConfigFile()
	env["AI_SDK_HARNESS_PI_TURN_FILE"] = input.layout.turnFile()
	env["PI_OFFLINE"] = "1"
	env["PI_SKIP_VERSION_CHECK"] = "1"
	env["PI_TELEMETRY"] = "0"

	if h.settings.CacheRetention == CacheRetentionLong {
		env["PI_CACHE_RETENTION"] = "long"
	}

	return env
}

// agentFileNames are the files copied from an explicit host agent
// directory.
var agentFileNames = []string{
	"models.json",
	"settings.json",
}

// readAgentFiles reads the model and general settings of an explicit host
// agent directory.
func (h *Harness) readAgentFiles() (map[string][]byte, error) {
	files := make(map[string][]byte)
	if h.settings.AgentDir == "" {
		return files, nil
	}

	for _, name := range agentFileNames {
		content, err := os.ReadFile(filepath.Join(h.settings.AgentDir, name))
		if errors.Is(err, fs.ErrNotExist) {
			continue
		}

		if err != nil {
			return nil, fmt.Errorf("read pi agent file %s: %w", name, err)
		}

		files[name] = content
	}

	return files, nil
}

// computeFingerprint identifies the launch configuration.
func (l launchConfiguration) computeFingerprint(thinkingLevel ThinkingLevel) (string, error) {
	encoded, err := json.Marshal(struct {
		Config        json.RawMessage   `json:"config"`
		Env           map[string]string `json:"env"`
		Extensions    []Extension       `json:"extensions"`
		AgentFiles    map[string][]byte `json:"agentFiles"`
		ThinkingLevel ThinkingLevel     `json:"thinkingLevel"`
		PiVersion     string            `json:"piVersion"`
	}{
		Config:        l.config,
		Env:           l.env,
		Extensions:    l.extensions,
		AgentFiles:    l.agentFiles,
		ThinkingLevel: thinkingLevel,
		PiVersion:     PiVersion,
	})
	if err != nil {
		return "", fmt.Errorf("fingerprint pi launch: %w", err)
	}

	sum := sha256.Sum256(encoded)

	return hex.EncodeToString(sum[:]), nil
}

// writeLaunchFiles writes the private directories, the extension
// configuration, an empty turn, the credential document, the agent files and
// the application's extensions.
func (s *session) writeLaunchFiles(ctx context.Context, launch launchConfiguration) error {
	directories := []string{
		s.layout.agentDir(),
		s.layout.sessionDir(),
		s.layout.extensionDir(),
	}

	quoted := make([]string, 0, len(directories))
	for _, directory := range directories {
		quoted = append(quoted, shellquote.Quote(directory))
	}

	err := bridgeadapter.RunChecked(ctx, s.sandbox, bridgeadapter.CheckedCommand{
		Command: "umask 077 && mkdir -p -- " + strings.Join(quoted, " "),
		Failure: "create pi session directories",
	})
	if err != nil {
		return err
	}

	files := map[string]string{
		s.layout.extensionConfigFile(): string(launch.config),
		s.layout.turnFile():            `{"tools":[]}`,
	}

	for name, content := range launch.agentFiles {
		files[path.Join(s.layout.agentDir(), name)] = string(content)
	}

	for _, extension := range launch.extensions {
		files[path.Join(s.layout.extensionDir(), extension.Name)] = extension.Content
	}

	for _, filePath := range slices.Sorted(maps.Keys(files)) {
		err := s.sandbox.WriteTextFile(ctx, sandbox.WriteTextFileOptions{
			Path:    filePath,
			Content: files[filePath],
		})
		if err != nil {
			return fmt.Errorf("write pi file %s: %w", path.Base(filePath), err)
		}
	}

	s.mu.Lock()
	s.toolsSignature = toolsSignature([]toolSpec{})
	s.mu.Unlock()

	return s.writeCredentialDocument(ctx)
}

// writeCredentialDocument writes the stored credential document as Pi's
// auth.json, readable only by its owner.
func (s *session) writeCredentialDocument(ctx context.Context) error {
	if s.credentials == nil || len(s.credentials.loaded) == 0 {
		return nil
	}

	target := path.Join(s.layout.agentDir(), "auth.json")

	err := bridgeadapter.RunChecked(ctx, s.sandbox, bridgeadapter.CheckedCommand{
		Command: "umask 077 && touch -- " + shellquote.Quote(target) + " && chmod 600 -- " + shellquote.Quote(target),
		Failure: "create pi credential file",
	})
	if err != nil {
		return err
	}

	err = s.sandbox.WriteBinaryFile(ctx, sandbox.WriteBinaryFileOptions{
		Path:    target,
		Content: s.credentials.loaded,
	})
	if err != nil {
		return fmt.Errorf("write pi credentials: %w", err)
	}

	// A sandbox may replace a file when it writes it, so the permissions are
	// restricted again.
	return bridgeadapter.RunChecked(ctx, s.sandbox, bridgeadapter.CheckedCommand{
		Command: "chmod 600 -- " + shellquote.Quote(target),
		Failure: "secure pi credential file",
	})
}

// launchArgs returns Pi's command-line arguments. Filesystem-discovered
// extensions, prompt templates and themes stay disabled, and project
// resources gated by project trust are ignored, as upstream loads only
// explicit extensions and isolated settings. Pi's built-in tools are
// replaced by the adapter's extension. The workspace's project skills are
// loaded explicitly, as upstream keeps them.
func (s *session) launchArgs(ctx context.Context, launch launchConfiguration) ([]string, error) {
	args := []string{
		"--mode", "rpc",
		"--session-dir", s.layout.sessionDir(),
		"--no-extensions",
		"--no-prompt-templates",
		"--no-themes",
		"--no-builtin-tools",
		"--no-approve",
		"--extension", s.layout.installFile(extensionScript),
	}

	if launch.usesMCP {
		args = append(args, "--extension", "builtin:mcp", "--extension", "builtin:tool-search")
	}

	for _, extension := range launch.extensions {
		args = append(args, "--extension", path.Join(s.layout.extensionDir(), extension.Name))
	}

	if level := s.harness.settings.ThinkingLevel; level != "" {
		args = append(args, "--thinking", string(level))
	}

	existing, err := s.existingPaths(ctx, []string{
		path.Join(s.layout.workDir, ".pi", "skills"),
		path.Join(s.layout.workDir, ".agents", "skills"),
	})
	if err != nil {
		return nil, err
	}

	for _, skillPath := range existing {
		args = append(args, "--skill", skillPath)
	}

	if s.sessionFileName != "" {
		sessionFile := s.layout.sessionFile(s.sessionFileName)

		resumable, err := s.existingPaths(ctx, []string{
			sessionFile,
		})
		if err != nil {
			return nil, err
		}

		if len(resumable) > 0 {
			args = append(args, "--session", sessionFile)
		}
	}

	return args, nil
}

// existingPaths returns the paths that exist in the sandbox.
func (s *session) existingPaths(ctx context.Context, candidates []string) ([]string, error) {
	var checks []string
	for _, candidate := range candidates {
		quoted := shellquote.Quote(candidate)
		checks = append(checks, fmt.Sprintf("if [ -e %s ]; then printf '%%s\\n' %s; fi", quoted, quoted))
	}

	result, err := s.sandbox.Run(ctx, sandbox.ProcessOptions{
		Command: strings.Join(checks, "; "),
	})
	if err != nil {
		return nil, fmt.Errorf("inspect pi session paths: %w", err)
	}

	var existing []string
	for line := range strings.SplitSeq(result.Stdout, "\n") {
		if slices.Contains(candidates, line) {
			existing = append(existing, line)
		}
	}

	return existing, nil
}
