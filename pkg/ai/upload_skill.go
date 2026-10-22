package ai

import (
	"context"
	"fmt"
	"net/http"

	"github.com/aarock1234/ai/pkg/model"
)

// UploadSkillOptions configures UploadSkill.
type UploadSkillOptions struct {
	// API is the skills interface, such as the result of a provider's
	// Skills method or Registry.Skills.
	API model.Skills

	// Files make up the skill, with paths relative to the skill root.
	Files []model.SkillFile

	// DisplayTitle is an optional human-readable title.
	DisplayTitle string

	Headers         http.Header
	ProviderOptions model.ProviderOptions

	// LogWarnings receives the warnings of the upload. Nil discards them;
	// they are always returned on the result.
	LogWarnings LogWarningsFunc
}

// UploadSkillResult is the outcome of UploadSkill.
type UploadSkillResult = model.UploadSkillResult

// UploadSkill uploads a skill through a provider's skills interface.
func UploadSkill(ctx context.Context, options UploadSkillOptions) (*UploadSkillResult, error) {
	if options.API == nil {
		return nil, &InvalidArgumentError{
			Parameter: "api",
			Message:   "api is required; the provider must offer a skills interface",
		}
	}

	result, err := options.API.UploadSkill(ctx, model.UploadSkillOptions{
		Files:           options.Files,
		DisplayTitle:    options.DisplayTitle,
		Headers:         options.Headers,
		ProviderOptions: options.ProviderOptions,
	})
	if err != nil {
		return nil, err
	}

	if result == nil {
		return nil, fmt.Errorf("%w: skills interface returned no result", model.ErrInvalidResponse)
	}

	options.LogWarnings.log(ctx, WarningsEvent{
		Warnings: result.Warnings,
		Provider: options.API.Provider(),
	})

	return result, nil
}
