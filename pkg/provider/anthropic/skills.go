package anthropic

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/urlpath"
	"github.com/aarock1234/ai/pkg/model"
)

// skillsBeta enables the Skills API.
const skillsBeta = "skills-2025-10-02"

var (
	_ model.Skills         = (*Skills)(nil)
	_ model.SkillsProvider = (*Provider)(nil)
)

// SkillMetadata is the provider metadata of an uploaded skill, stored under
// the "anthropic" namespace.
type SkillMetadata struct {
	Source    string `json:"source"`
	CreatedAt string `json:"createdAt"`
	UpdatedAt string `json:"updatedAt"`
}

// Skills uploads skills with the Anthropic Skills API.
type Skills struct {
	provider *Provider
}

// skillResponse is a skill object.
type skillResponse struct {
	ID            string  `json:"id"`
	DisplayTitle  *string `json:"display_title"`
	Name          *string `json:"name"`
	Description   *string `json:"description"`
	LatestVersion *string `json:"latest_version"`
	Source        *string `json:"source"`
	CreatedAt     *string `json:"created_at"`
	UpdatedAt     *string `json:"updated_at"`
}

// skillVersionResponse is a skill version object.
type skillVersionResponse struct {
	Type        *string `json:"type"`
	SkillID     *string `json:"skill_id"`
	Name        *string `json:"name"`
	Description *string `json:"description"`
}

// Provider returns the provider identifier, such as "anthropic.skills".
func (s *Skills) Provider() string {
	return strings.Replace(s.provider.name, ".messages", "", 1) + ".skills"
}

// UploadSkill uploads a skill from its files. The name and description are
// read from the latest version when the API reports one.
func (s *Skills) UploadSkill(ctx context.Context, options model.UploadSkillOptions) (*model.UploadSkillResult, error) {
	fields, err := skillFormFields(options)
	if err != nil {
		return nil, err
	}

	request, err := s.provider.betaRequest(ctx, "/skills", skillsBeta, options.Headers)
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

	if err := skill.validate(); err != nil {
		return nil, err
	}

	name := valueOrEmpty(skill.Name)
	description := valueOrEmpty(skill.Description)

	if latest := valueOrEmpty(skill.LatestVersion); latest != "" {
		version, err := s.version(ctx, skill.ID, latest, options.Headers)
		if err != nil {
			return nil, err
		}

		name = firstNonEmpty(valueOrEmpty(version.Name), name)
		description = firstNonEmpty(valueOrEmpty(version.Description), description)
	}

	return &model.UploadSkillResult{
		ProviderReference: model.ProviderReference{
			Namespace: skill.ID,
		},
		DisplayTitle:  valueOrEmpty(skill.DisplayTitle),
		Name:          name,
		Description:   description,
		LatestVersion: valueOrEmpty(skill.LatestVersion),
		ProviderMetadata: model.ProviderMetadata{
			Namespace: SkillMetadata{
				Source:    *skill.Source,
				CreatedAt: *skill.CreatedAt,
				UpdatedAt: *skill.UpdatedAt,
			},
		},
	}, nil
}

// version reads the metadata of one skill version with the upload's
// headers.
func (s *Skills) version(ctx context.Context, skillID string, version string, headers http.Header) (skillVersionResponse, error) {
	path := "/skills/" + urlpath.Escape(skillID) + "/versions/" + urlpath.Escape(version)

	request, err := s.provider.betaRequest(ctx, path, skillsBeta, headers)
	if err != nil {
		return skillVersionResponse{}, err
	}

	response, err := apicall.Get(ctx, request)
	if err != nil {
		return skillVersionResponse{}, err
	}

	var decoded skillVersionResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return skillVersionResponse{}, err
	}

	if decoded.Type == nil || decoded.SkillID == nil {
		return skillVersionResponse{}, fmt.Errorf("%w: skill version response has no type or skill_id", model.ErrInvalidResponse)
	}

	return decoded, nil
}

// skillFormFields returns the optional display title followed by one
// "files[]" field per skill file, named by its path.
func skillFormFields(options model.UploadSkillOptions) ([]apicall.FormField, error) {
	var fields []apicall.FormField

	if options.DisplayTitle != "" {
		fields = append(fields, apicall.FormField{
			Name:  "display_title",
			Value: options.DisplayTitle,
		})
	}

	for _, file := range options.Files {
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

// validate checks that the skill object has every required field.
func (r skillResponse) validate() error {
	isComplete := r.ID != "" &&
		r.Source != nil &&
		r.CreatedAt != nil &&
		r.UpdatedAt != nil

	if !isComplete {
		return fmt.Errorf("%w: skill response must have id, source, created_at and updated_at", model.ErrInvalidResponse)
	}

	return nil
}

func valueOrEmpty(value *string) string {
	if value == nil {
		return ""
	}

	return *value
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}

	return ""
}
