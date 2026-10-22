package harness

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"path"
	"strings"

	"github.com/aarock1234/ai/internal/shellquote"
	"github.com/aarock1234/ai/pkg/sandbox"
)

// instructionsMetadataVersion is the version of the instructions metadata
// file.
const instructionsMetadataVersion = 1

// WriteInstructionsOptions configure WriteInstructions.
type WriteInstructionsOptions struct {
	Sandbox sandbox.Session

	// HomePath is the absolute directory InstructionsFile is relative to.
	HomePath string

	// InstructionsFile is the relative path of the runtime's instructions
	// file, such as ".claude/CLAUDE.md".
	InstructionsFile string

	// Instructions are appended to the file's own content. Empty restores the
	// file to the content it had before the harness wrote to it.
	Instructions string
}

// WriteInstructionsResult reports what WriteInstructions did.
type WriteInstructionsResult struct {
	Changed  bool
	FilePath string
}

// instructionsMetadata records what the harness appended to an
// instructions file, so that it can update or remove exactly its own part.
type instructionsMetadata struct {
	Version         int     `json:"version"`
	OriginalContent *string `json:"originalContent"`
	Instructions    string  `json:"instructions"`
	AppliedContent  string  `json:"appliedContent"`
}

// WriteInstructions appends instructions to a runtime's instructions file,
// keeping the file's own content. A metadata file next to it records the
// applied instructions, so that a later call replaces them, and a call
// without instructions restores the original file, even when the user
// edited the file in between.
func WriteInstructions(ctx context.Context, options WriteInstructionsOptions) (WriteInstructionsResult, error) {
	targetPath, err := resolveHomeSubpath(options.HomePath, options.InstructionsFile, "instructionsFile")
	if err != nil {
		return WriteInstructionsResult{}, err
	}

	if strings.HasSuffix(options.InstructionsFile, "/") || strings.HasSuffix(options.InstructionsFile, "/.") {
		return WriteInstructionsResult{}, fmt.Errorf(
			"invalid instructionsFile %q: expected a relative POSIX path without traversal",
			options.InstructionsFile,
		)
	}

	metadataPath := path.Join(path.Dir(targetPath), "."+path.Base(targetPath)+".ai-sdk-harness-instructions.json")
	writer := instructionsWriter{
		session:      options.Sandbox,
		targetPath:   targetPath,
		metadataPath: metadataPath,
	}

	current, err := readOptionalText(ctx, options.Sandbox, targetPath)
	if err != nil {
		return WriteInstructionsResult{}, err
	}

	existing, err := writer.readMetadata(ctx)
	if err != nil {
		return WriteInstructionsResult{}, err
	}

	instructions := strings.TrimSpace(options.Instructions)
	if instructions != "" {
		return writer.apply(ctx, instructions, current, existing)
	}

	if existing == nil {
		return WriteInstructionsResult{
			FilePath: targetPath,
		}, nil
	}

	return writer.restore(ctx, current, *existing)
}

// instructionsWriter updates one instructions file and its metadata.
type instructionsWriter struct {
	session      sandbox.Session
	targetPath   string
	metadataPath string
}

// apply writes the file's own content followed by the instructions.
func (w instructionsWriter) apply(ctx context.Context, instructions string, current *string, existing *instructionsMetadata) (WriteInstructionsResult, error) {
	original := deriveOriginalContent(current, existing)

	target := instructions + "\n"
	if original != nil && strings.TrimSpace(*original) != "" {
		target = strings.TrimRight(*original, "\n") + "\n\n" + instructions + "\n"
	}

	unchanged := current != nil &&
		*current == target &&
		existing != nil &&
		existing.Instructions == instructions &&
		equalOptionalStrings(existing.OriginalContent, original)

	if unchanged {
		return WriteInstructionsResult{
			FilePath: w.targetPath,
		}, nil
	}

	err := w.session.WriteTextFile(ctx, sandbox.WriteTextFileOptions{
		Path:    w.targetPath,
		Content: target,
	})
	if err != nil {
		return WriteInstructionsResult{}, fmt.Errorf("write instructions file %s: %w", w.targetPath, err)
	}

	err = writeJSONAtomically(ctx, w.session, w.metadataPath, instructionsMetadata{
		Version:         instructionsMetadataVersion,
		OriginalContent: original,
		Instructions:    instructions,
		AppliedContent:  target,
	})
	if err != nil {
		return WriteInstructionsResult{}, err
	}

	return WriteInstructionsResult{
		Changed:  true,
		FilePath: w.targetPath,
	}, nil
}

// restore removes the harness's instructions from the file and deletes the
// metadata.
func (w instructionsWriter) restore(ctx context.Context, current *string, existing instructionsMetadata) (WriteInstructionsResult, error) {
	restored := deriveRestoredContent(current, existing)

	if restored != nil {
		err := w.session.WriteTextFile(ctx, sandbox.WriteTextFileOptions{
			Path:    w.targetPath,
			Content: strings.TrimRight(*restored, "\n") + "\n",
		})
		if err != nil {
			return WriteInstructionsResult{}, fmt.Errorf("restore instructions file %s: %w", w.targetPath, err)
		}
	} else {
		err := runSandboxCommand(ctx, w.session, "rm -f -- "+shellquote.Quote(w.targetPath), "remove instructions file "+w.targetPath)
		if err != nil {
			return WriteInstructionsResult{}, err
		}
	}

	err := runSandboxCommand(ctx, w.session, "rm -f -- "+shellquote.Quote(w.metadataPath), "remove instructions metadata "+w.metadataPath)
	if err != nil {
		return WriteInstructionsResult{}, err
	}

	return WriteInstructionsResult{
		Changed:  true,
		FilePath: w.targetPath,
	}, nil
}

// readMetadata reads the metadata, returning nil when it is missing.
func (w instructionsWriter) readMetadata(ctx context.Context) (*instructionsMetadata, error) {
	content, err := readOptionalText(ctx, w.session, w.metadataPath)
	if err != nil || content == nil {
		return nil, err
	}

	var metadata instructionsMetadata
	if err := json.Unmarshal([]byte(*content), &metadata); err != nil || metadata.Version != instructionsMetadataVersion {
		return nil, fmt.Errorf("invalid ai sdk harness instructions metadata: %s", w.metadataPath)
	}

	return &metadata, nil
}

// deriveOriginalContent recovers the file's own content: unchanged from the
// last write, with the user's edits before the appended instructions, or
// the whole file when the user replaced it.
func deriveOriginalContent(current *string, existing *instructionsMetadata) *string {
	if existing == nil {
		return current
	}

	if current == nil || *current == existing.AppliedContent {
		return existing.OriginalContent
	}

	if base, ok := userContentBefore(*current, existing.Instructions); ok {
		return base
	}

	if existing.OriginalContent != nil {
		return existing.OriginalContent
	}

	return current
}

// deriveRestoredContent returns the content to restore, or nil when the
// file should be removed.
func deriveRestoredContent(current *string, existing instructionsMetadata) *string {
	if current == nil {
		return nil
	}

	if *current == existing.AppliedContent {
		if existing.OriginalContent != nil && strings.TrimSpace(*existing.OriginalContent) != "" {
			return existing.OriginalContent
		}

		return nil
	}

	if base, ok := userContentBefore(*current, existing.Instructions); ok {
		return base
	}

	return current
}

// userContentBefore returns the content before the appended instructions.
// It reports false when the file does not end with them; a nil content means
// the file holds only the instructions.
func userContentBefore(current string, instructions string) (*string, bool) {
	trimmed := strings.TrimRight(current, "\n")

	if base, found := strings.CutSuffix(trimmed, "\n\n"+instructions); found {
		if base == "" {
			return nil, true
		}

		return &base, true
	}

	if trimmed == instructions {
		return nil, true
	}

	return nil, false
}

// readOptionalText reads a text file, returning nil when it is missing.
func readOptionalText(ctx context.Context, session sandbox.Session, filePath string) (*string, error) {
	content, err := session.ReadTextFile(ctx, sandbox.ReadTextFileOptions{
		Path: filePath,
	})
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}

	if err != nil {
		return nil, fmt.Errorf("read %s: %w", filePath, err)
	}

	return &content, nil
}

// equalOptionalStrings compares optional strings.
func equalOptionalStrings(first *string, second *string) bool {
	if first == nil || second == nil {
		return first == second
	}

	return *first == *second
}
