// Command devtools runs the AI SDK DevTools viewer on localhost, a port of
// upstream's npx @ai-sdk/devtools. Run it from the project directory whose
// calls are recorded; set AI_SDK_DEVTOOLS_PORT to use another port than
// 4983.
package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/aarock1234/ai/pkg/devtools"
)

func main() {
	if err := run(); err != nil {
		slog.Error("devtools failed", slog.Any("error", err))
		os.Exit(1)
	}
}

func run() error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	server, err := devtools.NewServer(devtools.ServerOptions{})
	if err != nil {
		return err
	}

	return server.ListenAndServe(ctx)
}
