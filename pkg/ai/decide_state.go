package ai

import (
	"context"
	"errors"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

var errUnexpectedDecisionFileConversion = errors.New("resolve decision state files: unexpected prompt conversion result")

// prepareDecisionState resolves the files of a decision state the way
// prompt files are resolved: URLs are downloaded, data URLs decoded and
// media types detected. Other parts are kept as they are, in order.
func prepareDecisionState(ctx context.Context, state []model.DecisionStatePart) ([]model.DecisionStatePart, error) {
	files := decisionStateFiles(state)
	if len(files) == 0 {
		return state, nil
	}

	resolved, err := resolveDecisionFiles(ctx, files)
	if err != nil {
		return nil, err
	}

	prepared := make([]model.DecisionStatePart, len(state))
	fileIndex := 0

	for index, part := range state {
		if _, isFile := part.(model.FilePart); !isFile {
			prepared[index] = part

			continue
		}

		prepared[index] = resolved[fileIndex]
		fileIndex++
	}

	return prepared, nil
}

func decisionStateFiles(state []model.DecisionStatePart) []model.UserPart {
	var files []model.UserPart

	for _, part := range state {
		if file, isFile := part.(model.FilePart); isFile {
			files = append(files, file)
		}
	}

	return files
}

// resolveDecisionFiles converts the files as one user message, for which
// no URL is supported by the model.
func resolveDecisionFiles(ctx context.Context, files []model.UserPart) ([]model.DecisionStatePart, error) {
	messages, err := convertToModelPrompt(ctx, convertPromptOptions{
		prompt: standardizedPrompt{
			messages: []model.Message{
				model.UserMessage{
					Content: files,
				},
			},
		},
	})
	if err != nil {
		return nil, err
	}

	if len(messages) != 1 {
		return nil, errUnexpectedDecisionFileConversion
	}

	message, ok := messages[0].(model.UserMessage)
	if !ok || len(message.Content) != len(files) {
		return nil, errUnexpectedDecisionFileConversion
	}

	resolved := make([]model.DecisionStatePart, len(message.Content))
	for index, part := range message.Content {
		file, isFile := part.(model.FilePart)
		if !isFile {
			return nil, fmt.Errorf("resolve decision state files: unexpected part %T", part)
		}

		resolved[index] = file
	}

	return resolved, nil
}
