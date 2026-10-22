package devtools

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

// shutdownTimeout bounds the graceful shutdown of the viewer.
const shutdownTimeout = 5 * time.Second

// maxNotifyBodyBytes bounds the body of a recorder notification.
const maxNotifyBodyBytes = 1 << 20

// viewerDevelopmentPort is the port of upstream's Vite development client,
// whose requests the API accepts.
const viewerDevelopmentPort = 5173

// ServerOptions configure NewServer.
type ServerOptions struct {
	// Port is the port to listen on. Zero uses the AI_SDK_DEVTOOLS_PORT
	// environment variable or DefaultPort.
	Port int

	// ProjectDirectory contains the .devtools directory to read until a
	// recorder names its database in a notification. Empty uses the
	// working directory.
	ProjectDirectory string

	// Logger receives server events. Nil uses slog.Default.
	Logger *slog.Logger
}

// Server is the DevTools viewer: a local HTTP server with upstream's React
// viewer, embedded from the published @ai-sdk/devtools build, and its JSON
// API. The API only accepts requests addressed to localhost on the server's
// port, so that other web pages cannot read the recordings.
//
// The viewer's assets are served under /assets/, and every other GET
// request receives the viewer page, which routes on the client. When the
// AI_SDK_DEVTOOLS_DEV environment variable is set to a value other than
// "false" and "0", the server only serves the API and every other page
// links to upstream's Vite development client on localhost:5173.
//
// The API is:
//
//	GET  /api/runs      root runs, most recent first, with stepCount, firstMessage, hasError, isInProgress and type
//	GET  /api/runs/{id} a run with its steps and, recursively, its child runs
//	POST /api/clear     deletes every run and step
//	GET  /api/events    server-sent "connected", "update" and "heartbeat" events
//	POST /api/notify    reloads the database; recorders call it after each change
type Server struct {
	port             int
	projectDirectory string
	logger           *slog.Logger
	events           *eventBroker
	mux              *http.ServeMux

	mu sync.Mutex

	// remoteDatabasePath is the database named by the latest notification,
	// so that the viewer works from any working directory.
	remoteDatabasePath string
}

// NewServer returns a viewer server.
func NewServer(options ServerOptions) (*Server, error) {
	projectDirectory, err := resolveProjectDirectory(options.ProjectDirectory)
	if err != nil {
		return nil, err
	}

	server := &Server{
		port:             resolvePort(options.Port),
		projectDirectory: projectDirectory,
		logger:           loggerOrDefault(options.Logger),
		events:           newEventBroker(),
		mux:              http.NewServeMux(),
	}

	server.mux.HandleFunc("GET /api/runs", server.handleRuns)
	server.mux.HandleFunc("GET /api/runs/{id}", server.handleRun)
	server.mux.HandleFunc("POST /api/clear", server.handleClear)
	server.mux.HandleFunc("GET /api/events", server.handleEvents)
	server.mux.HandleFunc("POST /api/notify", server.handleNotify)
	server.mux.HandleFunc("/", handleClient)

	return server, nil
}

// Port returns the port the server listens on.
func (s *Server) Port() int {
	return s.port
}

// ServeHTTP serves the viewer and its API.
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if isAPIPath(r.URL.Path) && !s.isAllowedRequest(r) {
		http.Error(w, "Forbidden", http.StatusForbidden)

		return
	}

	s.mux.ServeHTTP(w, r)
}

// ListenAndServe serves on localhost until ctx is cancelled, then shuts
// down gracefully. It returns an error when the port is in use, which
// usually means DevTools is already running.
func (s *Server) ListenAndServe(ctx context.Context) error {
	address := net.JoinHostPort("localhost", strconv.Itoa(s.port))

	listener, err := net.Listen("tcp", address)
	if err != nil {
		return fmt.Errorf("listen on %s (devtools may already be running; set %s to use another port): %w", address, portEnvironmentVariable, err)
	}

	requestCtx, cancelRequests := context.WithCancel(context.WithoutCancel(ctx))
	defer cancelRequests()

	server := &http.Server{
		Handler:           s,
		ReadHeaderTimeout: 10 * time.Second,
		BaseContext: func(net.Listener) context.Context {
			return requestCtx
		},
	}

	listenerResult := make(chan error, 1)
	go func() {
		listenerResult <- server.Serve(listener)
	}()

	s.logger.InfoContext(ctx, "devtools running", slog.String("url", "http://"+address))

	var listenErr error
	listenerFinished := false

	select {
	case err := <-listenerResult:
		listenErr = listenerError(err)
		listenerFinished = true
	case <-ctx.Done():
		s.logger.InfoContext(ctx, "shutting down devtools", slog.Any("cause", context.Cause(ctx)))
	}

	// Event streams run until their request context ends.
	cancelRequests()
	shutdownErr := shutdownServer(server)

	if !listenerFinished {
		listenErr = listenerError(<-listenerResult)
	}

	return errors.Join(listenErr, shutdownErr)
}

func (s *Server) handleClear(w http.ResponseWriter, r *http.Request) {
	store, err := OpenStore(StoreOptions{
		ProjectDirectory: s.viewedProjectDirectory(),
		Port:             s.port,
	})
	if err != nil {
		writeServerError(w, err)

		return
	}

	if err := store.Clear(r.Context()); err != nil {
		writeServerError(w, err)

		return
	}

	writeJSON(w, http.StatusOK, successResponse{
		Success: true,
	})
}

func (s *Server) handleNotify(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, maxNotifyBodyBytes)

	var body json.RawMessage
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "Invalid JSON body", http.StatusBadRequest)

		return
	}

	var notified struct {
		DBPath *string `json:"dbPath"`
	}
	if err := json.Unmarshal(body, &notified); err != nil {
		http.Error(w, "Invalid dbPath", http.StatusBadRequest)

		return
	}

	if notified.DBPath != nil {
		validated, ok := validateRemoteDatabasePath(*notified.DBPath)
		if !ok {
			http.Error(w, "Invalid dbPath", http.StatusBadRequest)

			return
		}

		s.mu.Lock()
		s.remoteDatabasePath = validated
		s.mu.Unlock()
	}

	var compacted bytes.Buffer
	if err := json.Compact(&compacted, body); err != nil {
		http.Error(w, "Invalid JSON body", http.StatusBadRequest)

		return
	}

	s.events.broadcast(serverEvent{
		name: "update",
		data: compacted.Bytes(),
	})

	writeJSON(w, http.StatusOK, successResponse{
		Success: true,
	})
}

// readViewedDatabase reads the database the viewer shows: the one named by
// the latest notification, or the project's.
func (s *Server) readViewedDatabase() *database {
	return readDatabase(s.viewedDatabasePath())
}

func (s *Server) viewedDatabasePath() string {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.remoteDatabasePath != "" {
		return s.remoteDatabasePath
	}

	return databasePath(s.projectDirectory)
}

func (s *Server) viewedProjectDirectory() string {
	return projectDirectoryOf(s.viewedDatabasePath())
}

// isAllowedRequest reports whether a request is addressed to this server on
// localhost and, when it comes from a web page, from the viewer itself.
func (s *Server) isAllowedRequest(r *http.Request) bool {
	if !s.isAllowedHost(r.Host) {
		return false
	}

	origin := r.Header.Get("Origin")

	return origin == "" || s.isAllowedOrigin(origin)
}

func (s *Server) isAllowedHost(host string) bool {
	for _, loopbackHost := range loopbackHosts() {
		if host == net.JoinHostPort(loopbackHost, strconv.Itoa(s.port)) {
			return true
		}
	}

	return false
}

func (s *Server) isAllowedOrigin(origin string) bool {
	for _, port := range []int{s.port, viewerDevelopmentPort} {
		for _, loopbackHost := range loopbackHosts() {
			if origin == "http://"+net.JoinHostPort(loopbackHost, strconv.Itoa(port)) {
				return true
			}
		}
	}

	return false
}

func loopbackHosts() []string {
	return []string{
		"localhost",
		"127.0.0.1",
		"::1",
	}
}

func isAPIPath(path string) bool {
	return strings.HasPrefix(path, "/api/")
}

type successResponse struct {
	Success bool `json:"success"`
}

type errorResponse struct {
	Error string `json:"error"`
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	encoded, err := json.Marshal(value)
	if err != nil {
		writeServerError(w, err)

		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(encoded)
}

func writeServerError(w http.ResponseWriter, err error) {
	http.Error(w, err.Error(), http.StatusInternalServerError)
}

func shutdownServer(server *http.Server) error {
	shutdownCtx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
	defer cancel()

	if err := server.Shutdown(shutdownCtx); err != nil {
		closeErr := server.Close()
		if closeErr != nil {
			closeErr = fmt.Errorf("close devtools server: %w", closeErr)
		}

		return errors.Join(fmt.Errorf("shut down devtools server: %w", err), closeErr)
	}

	return nil
}

func listenerError(err error) error {
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}

	return fmt.Errorf("serve devtools: %w", err)
}
