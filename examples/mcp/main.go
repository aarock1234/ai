// Command mcp starts an MCP server as a subprocess, connects to it over
// stdio, and lists the tools it exposes as ai.Tool values, ready to pass to
// ai.GenerateText.
//
// Usage:
//
//	go run ./examples/mcp pnpm dlx @modelcontextprotocol/server-everything
package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"time"

	"github.com/aarock1234/ai/pkg/mcp"
)

const closeTimeout = 5 * time.Second

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	if len(os.Args) < 2 {
		return errors.New("usage: mcp <server command> [arguments...]")
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	transport, err := mcp.NewStdioTransport(mcp.StdioTransportOptions{
		Command: os.Args[1],
		Args:    os.Args[2:],
	})
	if err != nil {
		return fmt.Errorf("start mcp server: %w", err)
	}

	client, err := mcp.NewClient(ctx, mcp.ClientOptions{
		Transport: transport,
	})
	if err != nil {
		return fmt.Errorf("connect to mcp server: %w", err)
	}
	defer closeClient(client)

	server := client.ServerInfo()
	fmt.Printf("connected to %s %s\n", server.Name, server.Version)

	tools, err := client.Tools(ctx, mcp.ToolsOptions{})
	if err != nil {
		return fmt.Errorf("list mcp tools: %w", err)
	}

	for _, tool := range tools {
		fmt.Printf("- %s: %s\n", tool.Name, tool.Description)
	}

	return nil
}

func closeClient(client *mcp.Client) {
	ctx, cancel := context.WithTimeout(context.Background(), closeTimeout)
	defer cancel()

	if err := client.Close(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "error: close mcp client: %v\n", err)
	}
}
