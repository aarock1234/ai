package openai

import (
	"encoding/json"
	"fmt"
	"regexp"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// responsesImageSizePattern matches custom WIDTHxHEIGHT image sizes.
var responsesImageSizePattern = regexp.MustCompile(`^\d+x\d+$`)

// addProviderTool adds an OpenAI tool. Tools of other providers are
// ignored.
func (p *responsesToolPreparer) addProviderTool(tool model.ProviderTool) error {
	converted, err := p.convertProviderTool(tool)
	if err != nil {
		return err
	}

	if converted == nil {
		return nil
	}

	p.prepared.tools = append(p.prepared.tools, converted)
	p.record(tool.Name, responsesAllowedResolution(converted), p.request.toolNames.ToProvider(tool.Name))

	return nil
}

func (p *responsesToolPreparer) convertProviderTool(tool model.ProviderTool) (responsesTool, error) {
	switch tool.ID {
	case fileSearchToolID:
		return convertResponsesTool(tool, responsesFileSearch)
	case localShellToolID:
		return responsesBuiltinTool{
			Type: "local_shell",
		}, nil
	case shellToolID:
		return p.shellTool(tool)
	case applyPatchToolID:
		return responsesBuiltinTool{
			Type: "apply_patch",
		}, nil
	case computerToolID:
		return responsesBuiltinTool{
			Type: "computer",
		}, nil
	case webSearchPreviewToolID:
		return convertResponsesTool(tool, responsesWebSearchPreview)
	case webSearchToolID:
		return convertResponsesTool(tool, responsesWebSearch)
	case codeInterpreterToolID:
		return convertResponsesTool(tool, responsesCodeInterpreter)
	case imageGenerationToolID:
		return convertResponsesTool(tool, responsesImageGeneration)
	case mcpToolID:
		return convertResponsesTool(tool, responsesMCP)
	case customToolID:
		return p.customTool(tool)
	case programmaticToolCallingToolID:
		return responsesBuiltinTool{
			Type: "programmatic_tool_calling",
		}, nil
	case toolSearchToolID:
		return convertResponsesTool(tool, responsesToolSearch)
	default:
		return nil, nil
	}
}

func (p *responsesToolPreparer) shellTool(tool model.ProviderTool) (responsesTool, error) {
	args, err := responsesDecodeToolArgs[ShellArgs](tool)
	if err != nil {
		return nil, err
	}

	// The first shell tool decides who executes shell calls.
	isFirstShell := !p.shellSeen
	p.shellSeen = true

	shell := responsesShellTool{
		Type: "shell",
	}

	if args.Environment == nil {
		return shell, nil
	}

	environment, err := responsesShellEnvironmentOf(*args.Environment)
	if err != nil {
		return nil, err
	}

	shell.Environment = &environment

	if isFirstShell {
		environmentType := args.Environment.Type
		p.prepared.shellProviderExecuted = environmentType == ShellContainerAuto || environmentType == ShellContainerReference
	}

	return shell, nil
}

func (p *responsesToolPreparer) customTool(tool model.ProviderTool) (responsesTool, error) {
	args, err := responsesDecodeToolArgs[CustomToolArgs](tool)
	if err != nil {
		return nil, err
	}

	if args.Format != nil {
		if err := responsesCheckEnum("custom tool format", args.Format.Type, "grammar", "text"); err != nil {
			return nil, err
		}

		if err := responsesCheckEnum("custom tool syntax", args.Format.Syntax, "regex", "lark"); err != nil {
			return nil, err
		}
	}

	p.prepared.customToolNames[tool.Name] = true

	return responsesCustomTool{
		Type:        "custom",
		Name:        tool.Name,
		Description: args.Description,
		Async:       p.asyncOption(args.Async, tool.Name),
		Format:      args.Format,
	}, nil
}

// convertResponsesTool decodes a provider tool's arguments and converts
// them.
func convertResponsesTool[Args any](tool model.ProviderTool, convert func(Args) (responsesTool, error)) (responsesTool, error) {
	args, err := responsesDecodeToolArgs[Args](tool)
	if err != nil {
		return nil, err
	}

	return convert(args)
}

// responsesDecodeToolArgs decodes the arguments of a provider tool.
func responsesDecodeToolArgs[Args any](tool model.ProviderTool) (Args, error) {
	var args Args
	if len(tool.Args) == 0 {
		return args, nil
	}

	if err := json.Unmarshal(tool.Args, &args); err != nil {
		return args, fmt.Errorf("%w: %s tool arguments: %w", model.ErrInvalidArgument, tool.ID, err)
	}

	return args, nil
}

func responsesFileSearch(args FileSearchArgs) (responsesTool, error) {
	tool := responsesFileSearchTool{
		Type:           "file_search",
		VectorStoreIDs: args.VectorStoreIDs,
		MaxNumResults:  args.MaxNumResults,
		Filters:        args.Filters,
	}

	if tool.VectorStoreIDs == nil {
		tool.VectorStoreIDs = []string{}
	}

	if args.Ranking != nil {
		tool.RankingOptions = &responsesRankingOptions{
			Ranker:         args.Ranking.Ranker,
			ScoreThreshold: args.Ranking.ScoreThreshold,
		}
	}

	return tool, nil
}

func responsesWebSearchPreview(args WebSearchPreviewArgs) (responsesTool, error) {
	if err := responsesCheckEnum("searchContextSize", args.SearchContextSize, "low", "medium", "high"); err != nil {
		return nil, err
	}

	return responsesWebSearchTool{
		Type:              "web_search_preview",
		SearchContextSize: args.SearchContextSize,
		UserLocation:      args.UserLocation,
	}, nil
}

func responsesWebSearch(args WebSearchArgs) (responsesTool, error) {
	if err := responsesCheckEnum("searchContextSize", args.SearchContextSize, "low", "medium", "high"); err != nil {
		return nil, err
	}

	tool := responsesWebSearchTool{
		Type:              "web_search",
		ExternalWebAccess: args.ExternalWebAccess,
		SearchContextSize: args.SearchContextSize,
		UserLocation:      args.UserLocation,
	}

	if args.Filters != nil {
		tool.Filters = &responsesDomainFilters{
			AllowedDomains: args.Filters.AllowedDomains,
			BlockedDomains: args.Filters.BlockedDomains,
		}
	}

	return tool, nil
}

func responsesCodeInterpreter(args CodeInterpreterArgs) (responsesTool, error) {
	tool := responsesCodeInterpreterTool{
		Type: "code_interpreter",
	}

	if args.Container != nil {
		tool.Container = responsesCodeInterpreterContainer{
			ID:      args.Container.ID,
			FileIDs: args.Container.FileIDs,
		}
	}

	return tool, nil
}

func responsesImageGeneration(args ImageGenerationArgs) (responsesTool, error) {
	if err := responsesCheckImageGenerationArgs(args); err != nil {
		return nil, err
	}

	tool := responsesImageGenerationTool{
		Type:              "image_generation",
		Action:            args.Action,
		Background:        args.Background,
		InputFidelity:     args.InputFidelity,
		Model:             args.Model,
		Moderation:        args.Moderation,
		OutputCompression: args.OutputCompression,
		OutputFormat:      args.OutputFormat,
		PartialImages:     args.PartialImages,
		Quality:           args.Quality,
		Size:              args.Size,
	}

	if args.InputImageMask != nil {
		tool.InputImageMask = &responsesInputImageMask{
			FileID:   args.InputImageMask.FileID,
			ImageURL: args.InputImageMask.ImageURL,
		}
	}

	return tool, nil
}

// responsesCheckImageGenerationArgs validates the enumerations and ranges
// of image generation arguments.
func responsesCheckImageGenerationArgs(args ImageGenerationArgs) error {
	checks := []error{
		responsesCheckEnum("image generation action", args.Action, "generate", "edit", "auto"),
		responsesCheckEnum("image generation background", args.Background, "auto", "opaque", "transparent"),
		responsesCheckEnum("image generation input fidelity", args.InputFidelity, "low", "high"),
		responsesCheckEnum("image generation moderation", args.Moderation, "auto", "low"),
		responsesCheckEnum("image generation output format", args.OutputFormat, "png", "jpeg", "webp"),
		responsesCheckEnum("image generation quality", args.Quality, "auto", "low", "medium", "high", "xhigh", "max"),
	}

	for _, err := range checks {
		if err != nil {
			return err
		}
	}

	if compression := args.OutputCompression; compression != nil && (*compression < 0 || *compression > 100) {
		return fmt.Errorf("%w: image generation output compression must be between 0 and 100", model.ErrInvalidArgument)
	}

	if partial := args.PartialImages; partial != nil && (*partial < 0 || *partial > 3) {
		return fmt.Errorf("%w: image generation partial images must be between 0 and 3", model.ErrInvalidArgument)
	}

	if args.Size != "" && args.Size != "auto" && !responsesImageSizePattern.MatchString(args.Size) {
		return fmt.Errorf("%w: unsupported image generation size %q", model.ErrInvalidArgument, args.Size)
	}

	return nil
}

func responsesMCP(args MCPArgs) (responsesTool, error) {
	if args.ServerURL == "" && args.ConnectorID == "" {
		return nil, fmt.Errorf("%w: one of serverUrl or connectorId must be provided", model.ErrInvalidArgument)
	}

	tool := responsesMCPTool{
		Type:              "mcp",
		ServerLabel:       args.ServerLabel,
		Authorization:     args.Authorization,
		ConnectorID:       args.ConnectorID,
		Headers:           args.Headers,
		RequireApproval:   responsesMCPApprovalOf(args.RequireApproval),
		ServerDescription: args.ServerDescription,
		ServerURL:         args.ServerURL,
	}

	if allowed := args.AllowedTools; allowed != nil {
		tool.AllowedTools = &responsesMCPAllowed{
			ToolNames: allowed.ToolNames,
			Filter:    responsesMCPFilterOf(allowed.Filter),
		}
	}

	return tool, nil
}

// responsesMCPApprovalOf converts the approval policy. Without a policy the
// server never requires approval.
func responsesMCPApprovalOf(approval *MCPApproval) responsesMCPApproval {
	never := responsesMCPApproval{
		Policy: "never",
	}

	if approval == nil {
		return never
	}

	if approval.Policy != "" {
		return responsesMCPApproval{
			Policy: approval.Policy,
		}
	}

	if approval.Never == nil {
		return never
	}

	return responsesMCPApproval{
		Never: &responsesMCPFilter{
			ToolNames: approval.Never.ToolNames,
		},
	}
}

func responsesMCPFilterOf(filter *MCPToolFilter) *responsesMCPFilter {
	if filter == nil {
		return nil
	}

	return &responsesMCPFilter{
		ReadOnly:  filter.ReadOnly,
		ToolNames: filter.ToolNames,
	}
}

func responsesToolSearch(args ToolSearchArgs) (responsesTool, error) {
	if err := responsesCheckEnum("tool search execution", args.Execution, "server", "client"); err != nil {
		return nil, err
	}

	if len(args.Parameters) > 0 && !jsonobject.IsObject(args.Parameters) {
		return nil, fmt.Errorf("%w: tool search parameters must be a JSON Schema object", model.ErrInvalidArgument)
	}

	return responsesToolSearchTool{
		Type:        "tool_search",
		Execution:   args.Execution,
		Description: args.Description,
		Parameters:  args.Parameters,
	}, nil
}

// responsesShellEnvironmentOf converts a shell environment.
func responsesShellEnvironmentOf(environment ShellEnvironment) (responsesShellEnvironment, error) {
	switch environment.Type {
	case ShellContainerReference:
		return responsesShellEnvironment{
			Type:        "container_reference",
			ContainerID: environment.ContainerID,
		}, nil

	case ShellContainerAuto:
		skills, err := responsesContainerSkills(environment.Skills)
		if err != nil {
			return responsesShellEnvironment{}, err
		}

		return responsesShellEnvironment{
			Type:          "container_auto",
			FileIDs:       environment.FileIDs,
			MemoryLimit:   environment.MemoryLimit,
			NetworkPolicy: responsesNetworkPolicyOf(environment.NetworkPolicy),
			Skills:        skills,
		}, nil

	case ShellLocal, "":
		return responsesShellEnvironment{
			Type:   "local",
			Skills: responsesLocalSkills(environment.Skills),
		}, nil

	default:
		return responsesShellEnvironment{}, fmt.Errorf("%w: unsupported shell environment %q", model.ErrInvalidArgument, environment.Type)
	}
}

func responsesNetworkPolicyOf(policy *ShellNetworkPolicy) *responsesNetworkPolicy {
	if policy == nil {
		return nil
	}

	if policy.Type == "disabled" {
		return &responsesNetworkPolicy{
			Type: "disabled",
		}
	}

	return &responsesNetworkPolicy{
		Type:           "allowlist",
		AllowedDomains: policy.AllowedDomains,
		DomainSecrets:  policy.DomainSecrets,
	}
}

// responsesContainerSkills converts container skills. Skill references
// resolve the "openai" skill ID and default to the latest version.
func responsesContainerSkills(skills []ShellSkill) ([]responsesShellSkill, error) {
	var converted []responsesShellSkill

	for _, skill := range skills {
		if skill.Type == "skillReference" {
			skillID, ok := skill.ProviderReference[Namespace]
			if !ok {
				return nil, fmt.Errorf("%w: shell skill reference has no openai skill ID", model.ErrInvalidArgument)
			}

			version := skill.Version
			if version == "" {
				version = "latest"
			}

			converted = append(converted, responsesShellSkill{
				Type:    "skill_reference",
				SkillID: skillID,
				Version: version,
			})

			continue
		}

		if skill.Source == nil {
			return nil, fmt.Errorf("%w: inline shell skill %q has no source", model.ErrInvalidArgument, skill.Name)
		}

		converted = append(converted, responsesShellSkill{
			Type:        "inline",
			Name:        skill.Name,
			Description: skill.Description,
			Source: &responsesShellSkillSource{
				Type:      "base64",
				MediaType: skill.Source.MediaType,
				Data:      skill.Source.Data,
			},
		})
	}

	return converted, nil
}

// responsesLocalSkills converts the skills of a local environment.
func responsesLocalSkills(skills []ShellSkill) []responsesShellSkill {
	var converted []responsesShellSkill

	for _, skill := range skills {
		converted = append(converted, responsesShellSkill{
			Name:        skill.Name,
			Description: skill.Description,
			Path:        skill.Path,
		})
	}

	return converted
}
