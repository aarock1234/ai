package model

import (
	"context"
	"net/http"
)

// Skills uploads agent skills, bundles of instructions and files that
// models load on demand, to a provider.
type Skills interface {
	// Provider returns the provider identifier, such as "openai.skills".
	Provider() string

	// UploadSkill uploads a new skill from its files.
	UploadSkill(ctx context.Context, options UploadSkillOptions) (*UploadSkillResult, error)
}

// SkillsProvider is implemented by providers that offer a skills interface.
// It is separate from Provider because most providers have none.
type SkillsProvider interface {
	Skills() Skills
}

// SkillFile is one file of a skill.
type SkillFile struct {
	// Path is relative to the skill root, such as "my-skill/SKILL.md".
	Path string

	Data InlineFileData
}

// UploadSkillOptions are the inputs of one skill upload.
type UploadSkillOptions struct {
	Files []SkillFile

	// DisplayTitle is an optional human-readable title.
	DisplayTitle string

	// Headers are added to the request. Upstream's skills interface takes
	// no headers; the Go port accepts them like the files interface does.
	Headers         http.Header
	ProviderOptions ProviderOptions
}

// UploadSkillResult is the outcome of a skill upload. Empty fields were not
// reported by the provider.
type UploadSkillResult struct {
	// ProviderReference maps the provider name to the skill ID.
	ProviderReference ProviderReference

	DisplayTitle string
	Name         string
	Description  string

	// LatestVersion identifies the newest version of the skill.
	LatestVersion string

	ProviderMetadata ProviderMetadata
	Warnings         []Warning
}
