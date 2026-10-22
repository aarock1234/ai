// Command chat-server serves a chat endpoint for the useChat hook of the AI
// SDK UI. It converts the UI messages it receives into a model prompt,
// streams the model's answer and writes it back as a UI message stream.
//
// Usage:
//
//	OPENAI_API_KEY=... go run ./examples/chat-server
//
// Then point useChat at the endpoint:
//
//	useChat({
//	  transport: new DefaultChatTransport({
//	    api: "http://localhost:8080/api/chat",
//	  }),
//	})
package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/aarock1234/ai/pkg/provider/openai"
)

const (
	listenAddress     = "localhost:8080"
	readHeaderTimeout = 10 * time.Second
	shutdownTimeout   = 10 * time.Second
)

func main() {
	if err := run(); err != nil {
		slog.Error("fatal error", slog.Any("error", err))
		os.Exit(1)
	}
}

func run() error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	provider := openai.New()

	chat := &chatHandler{
		model: provider.Responses("gpt-5.4"),
	}

	router := http.NewServeMux()
	router.Handle("POST /api/chat", chat)

	// Streaming responses outlive any fixed write timeout, so only reading
	// the request headers is bounded.
	server := &http.Server{
		Addr:              listenAddress,
		Handler:           router,
		ReadHeaderTimeout: readHeaderTimeout,
	}

	return serve(ctx, server)
}

// serve runs the server until ctx is cancelled, then shuts it down. A listener
// or shutdown failure is returned even when it happens during cancellation.
func serve(ctx context.Context, server *http.Server) error {
	var listenConfig net.ListenConfig

	listener, err := listenConfig.Listen(ctx, "tcp", server.Addr)
	if err != nil {
		return fmt.Errorf("listen: %w", err)
	}

	slog.InfoContext(ctx, "server listening", slog.String("addr", listener.Addr().String()))

	serveResult := make(chan error, 1)

	go func() {
		serveResult <- server.Serve(listener)
	}()

	select {
	case err := <-serveResult:
		return serverError(err)
	case <-ctx.Done():
		slog.InfoContext(ctx, "shutting down server")
	}

	shutdownErr := shutdownServer(server)
	serveErr := serverError(<-serveResult)

	return errors.Join(serveErr, shutdownErr)
}

// shutdownServer stops the server gracefully, closing open connections when
// the timeout expires first. It uses a fresh context because the process
// context is already cancelled.
func shutdownServer(server *http.Server) error {
	ctx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		closeErr := server.Close()
		if closeErr != nil {
			closeErr = fmt.Errorf("close server: %w", closeErr)
		}

		return errors.Join(fmt.Errorf("shutdown server: %w", err), closeErr)
	}

	return nil
}

// serverError treats the result of a shut down server as success.
func serverError(err error) error {
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}

	return fmt.Errorf("serve: %w", err)
}
