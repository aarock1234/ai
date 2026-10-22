// Package bash is a provider-independent bash tool for any model with
// function calling, the Go port of upstream @ai-sdk/tool-bash. Commands run
// in a supplied sandbox session, such as sandbox/local or sandbox/vercel;
// the tool has no dependency on either.
//
// The caller owns the sandbox's lifecycle. Tools sharing a session share its
// files. Available commands, networking and persistence depend on the
// sandbox.
package bash

import (
	"context"
	"fmt"
	"path"

	"github.com/aarock1234/ai/internal/shellquote"
	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/sandbox"
)

// ToolName is the name of the tool.
const ToolName = "bash"

// defaultMaxOutputLength is the default limit of each output stream.
const defaultMaxOutputLength = 30_000

// Options configure New.
type Options struct {
	// Sandbox runs the commands. It is required.
	Sandbox sandbox.Session

	// WorkingDirectory is an absolute sandbox directory every command starts
	// in; New creates it. Empty uses the sandbox's default directory and
	// runs no setup command.
	WorkingDirectory string

	// MaxOutputLength is the number of characters kept from each output
	// stream before a truncation notice. Zero selects 30,000.
	MaxOutputLength int
}

// Input is the input of the tool.
type Input struct {
	// Command is the Bash command or script to execute.
	Command string `json:"command" jsonschema:"description=The Bash command or script to execute."`
}

// Output is the output of the tool.
type Output struct {
	Stdout   string `json:"stdout"`
	Stderr   string `json:"stderr"`
	ExitCode int    `json:"exitCode"`
}

// New returns the bash tool. Cancellation of a call is forwarded to the
// sandbox, which kills the command.
func New(ctx context.Context, options Options) (ai.Tool, error) {
	if options.Sandbox == nil {
		return ai.Tool{}, &ai.InvalidArgumentError{
			Parameter: "Sandbox",
			Message:   "a sandbox session is required",
		}
	}

	maxOutputLength := options.MaxOutputLength
	if maxOutputLength == 0 {
		maxOutputLength = defaultMaxOutputLength
	}

	if maxOutputLength < 1 {
		return ai.Tool{}, &ai.InvalidArgumentError{
			Parameter: "MaxOutputLength",
			Value:     options.MaxOutputLength,
			Message:   "must be a positive integer",
		}
	}

	workingDirectory, err := prepareWorkingDirectory(ctx, options)
	if err != nil {
		return ai.Tool{}, err
	}

	runner := commandRunner{
		sandbox:          options.Sandbox,
		workingDirectory: workingDirectory,
		maxOutputLength:  maxOutputLength,
	}

	tool := ai.NewTool(ai.ToolDefinition[Input, Output]{
		Name:        ToolName,
		Description: description(workingDirectory, options.Sandbox.Description()),
		Execute: func(ctx context.Context, input Input, _ ai.ToolExecutionOptions) (Output, error) {
			return runner.run(ctx, input.Command)
		},
	})

	return tool, nil
}

// prepareWorkingDirectory validates the working directory and creates it.
func prepareWorkingDirectory(ctx context.Context, options Options) (string, error) {
	if options.WorkingDirectory == "" {
		return "", nil
	}

	if !path.IsAbs(options.WorkingDirectory) {
		return "", &ai.InvalidArgumentError{
			Parameter: "WorkingDirectory",
			Value:     options.WorkingDirectory,
			Message:   "must be an absolute path in the sandbox",
		}
	}

	workingDirectory := path.Clean(options.WorkingDirectory)

	result, err := options.Sandbox.Run(ctx, sandbox.ProcessOptions{
		Command: "mkdir -p -- " + shellquote.Quote(workingDirectory),
	})
	if err != nil {
		return "", fmt.Errorf("create working directory: %w", err)
	}

	if result.ExitCode != 0 {
		return "", fmt.Errorf("create working directory: %s", result.Stderr)
	}

	return workingDirectory, nil
}

// commandRunner runs the tool's commands.
type commandRunner struct {
	sandbox          sandbox.Session
	workingDirectory string
	maxOutputLength  int
}

// run runs one command and limits its output.
func (r commandRunner) run(ctx context.Context, command string) (Output, error) {
	result, err := r.sandbox.Run(ctx, sandbox.ProcessOptions{
		Command:          command,
		WorkingDirectory: r.workingDirectory,
	})
	if err != nil {
		return Output{}, err
	}

	return Output{
		Stdout:   limitOutput(result.Stdout, r.maxOutputLength),
		Stderr:   limitOutput(result.Stderr, r.maxOutputLength),
		ExitCode: result.ExitCode,
	}, nil
}

// description describes the tool to the model.
func description(workingDirectory string, sandboxDescription string) string {
	location := "Commands run in the sandbox's default working directory. "
	if workingDirectory != "" {
		location = fmt.Sprintf("Every command starts in %s. ", workingDirectory)
	}

	return "Run Bash commands in the sandbox. " +
		location +
		"Use ls to discover files and commands such as cat, grep, sort, awk, " +
		"and jq to inspect and transform them. Pipes and redirections are supported. " +
		"Returns stdout, stderr, and exitCode; long output is truncated.\n" +
		sandboxDescription
}

// limitOutput keeps the first maxLength characters of output and appends a
// truncation notice.
func limitOutput(output string, maxLength int) string {
	characters := []rune(output)
	if len(characters) <= maxLength {
		return output
	}

	return fmt.Sprintf(
		"%s\n[Output truncated: %d characters omitted.]",
		string(characters[:maxLength]),
		len(characters)-maxLength,
	)
}
