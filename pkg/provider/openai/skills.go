package openai

import (
	"context"
	"fmt"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

var (
	_ model.Skills         = (*Skills)(nil)
	_ model.SkillsProvider = (*Provider)(nil)
)

// SkillMetadata is the provider metadata of an uploaded skill, stored under
// the "openai" namespace. Fields OpenAI did not report are omitted.
type SkillMetadata struct {
	DefaultVersion string `json:"defaultVersion,omitempty"`
	CreatedAt      *int64 `json:"createdAt,omitempty"`
	UpdatedAt      *int64 `json:"updatedAt,omitempty"`
}

// Skills uploads skills with the OpenAI Skills API.
type Skills struct {
	config config
}

// skillResponse is a skill object.
type skillResponse struct {
	ID             string  `json:"id"`
	Name           *string `json:"name"`
	Description    *string `json:"description"`
	DefaultVersion *string `json:"default_version"`
	LatestVersion  *string `json:"latest_version"`
	CreatedAt      *int64  `json:"created_at"`
	UpdatedAt      *int64  `json:"updated_at"`
}

// Provider returns the provider identifier, such as "openai.skills".
func (s *Skills) Provider() string {
	return s.config.modelProvider("skills")
}

// UploadSkill uploads a skill from its files. OpenAI has no display title,
// so a title produces an unsupported warning.
func (s *Skills) UploadSkill(ctx context.Context, options model.UploadSkillOptions) (*model.UploadSkillResult, error) {
	var warnings []model.Warning
	if options.DisplayTitle != "" {
		warnings = append(warnings, model.UnsupportedWarning("displayTitle", ""))
	}

	fields, err := skillFileFields(options.Files)
	if err != nil {
		return nil, err
	}

	request, err := s.config.request(ctx, "/skills", options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostForm(ctx, request, fields)
	if err != nil {
		return nil, err
	}

	var skill skillResponse
	if err := apicall.DecodeJSON(response.Body, &skill); err != nil {
		return nil, err
	}

	if skill.ID == "" || skill.CreatedAt == nil {
		return nil, fmt.Errorf("%w: skill response has no id or created_at", model.ErrInvalidResponse)
	}

	return &model.UploadSkillResult{
		ProviderReference: model.ProviderReference{
			Namespace: skill.ID,
		},
		Name:          valueOr(skill.Name, ""),
		Description:   valueOr(skill.Description, ""),
		LatestVersion: valueOr(skill.LatestVersion, ""),
		ProviderMetadata: model.ProviderMetadata{
			Namespace: SkillMetadata{
				DefaultVersion: valueOr(skill.DefaultVersion, ""),
				CreatedAt:      skill.CreatedAt,
				UpdatedAt:      skill.UpdatedAt,
			},
		},
		Warnings: warnings,
	}, nil
}

// skillFileFields returns one "files[]" form field per skill file, named by
// its path.
func skillFileFields(files []model.SkillFile) ([]apicall.FormField, error) {
	fields := make([]apicall.FormField, 0, len(files))

	for _, file := range files {
		content, err := model.InlineFileBytes(file.Data)
		if err != nil {
			return nil, err
		}

		fields = append(fields, apicall.FormField{
			Name:     "files[]",
			Filename: file.Path,
			Data:     content,
		})
	}

	return fields, nil
}
