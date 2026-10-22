package google

import (
	"fmt"
	"strings"
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

// defaultInteractionPollTimeout bounds the polling of a background
// interaction. Agent runs such as deep research take tens of minutes.
const defaultInteractionPollTimeout = 30 * time.Minute

// preparedInteraction is a converted Interactions API call.
type preparedInteraction struct {
	body     interactionsRequest
	warnings []model.Warning

	isAgent      bool
	isBackground bool
	pollTimeout  time.Duration
}

// prepareInteraction converts call options. Agent calls take an agent
// configuration and an environment instead of the generation config,
// structured output and generation settings, which are dropped with a
// warning.
func (m *InteractionsModel) prepareInteraction(options model.CallOptions) (preparedInteraction, error) {
	googleOptions, err := readInteractionsOptions(options.ProviderOptions)
	if err != nil {
		return preparedInteraction{}, err
	}

	isAgent := m.agent != ""

	var warnings []model.Warning

	if !isAgent {
		warnings = append(warnings, penaltyWarnings(options)...)
	}

	tools, err := prepareInteractionsTools(options.Tools, options.ToolChoice)
	if err != nil {
		return preparedInteraction{}, err
	}

	warnings = append(warnings, tools.warnings...)

	responseFormat, formatWarnings := interactionsResponseFormats(options.ResponseFormat, googleOptions, isAgent)
	warnings = append(warnings, formatWarnings...)

	input, err := convertInteractionsInput(options.Prompt, interactionsInputSettings{
		previousInteractionID: valueOrEmpty(googleOptions.PreviousInteractionID),
		store:                 googleOptions.Store,
		mediaResolution:       googleOptions.MediaResolution,
	})
	if err != nil {
		return preparedInteraction{}, err
	}

	warnings = append(warnings, input.warnings...)

	systemInstruction, systemWarnings := interactionsSystemInstruction(input, googleOptions.SystemInstruction)
	warnings = append(warnings, systemWarnings...)

	body := interactionsRequest{
		Input:                 input.input,
		SystemInstruction:     systemInstruction,
		Tools:                 tools.tools,
		ResponseFormat:        responseFormat,
		ResponseModalities:    googleOptions.ResponseModalities,
		PreviousInteractionID: valueOrEmpty(googleOptions.PreviousInteractionID),
		ServiceTier:           googleOptions.ServiceTier,
		Store:                 googleOptions.Store,
		Background:            googleOptions.Background,
	}

	if isAgent {
		body.Agent = m.agent
		body.AgentConfig = interactionsAgentConfigOf(googleOptions.AgentConfig)
		body.Environment = googleOptions.Environment

		warnings = append(warnings, droppedAgentFieldsWarning(options, googleOptions)...)
	} else {
		body.Model = m.modelID
		body.GenerationConfig = interactionsGenerationConfigOf(options, googleOptions, tools.toolChoice)

		if googleOptions.Environment != nil {
			warnings = append(warnings, otherWarning(
				"google.interactions: environment is only supported when an agent is set; environment will be omitted from the request body.",
			))
		}
	}

	pollTimeout := defaultInteractionPollTimeout
	if googleOptions.PollingTimeoutMs != nil {
		pollTimeout = time.Duration(*googleOptions.PollingTimeoutMs) * time.Millisecond
	}

	return preparedInteraction{
		body:         body,
		warnings:     warnings,
		isAgent:      isAgent,
		isBackground: googleOptions.Background != nil && *googleOptions.Background,
		pollTimeout:  pollTimeout,
	}, nil
}

// interactionsSystemInstruction prefers the prompt's system messages over
// the systemInstruction option.
func interactionsSystemInstruction(input convertedInteractionsInput, option *string) (string, []model.Warning) {
	if !input.hasSystem {
		return valueOrEmpty(option), nil
	}

	if option == nil {
		return input.systemInstruction, nil
	}

	return input.systemInstruction, []model.Warning{
		otherWarning("google.interactions: both AI SDK system message and providerOptions.google.systemInstruction were set; using the AI SDK system message."),
	}
}

// interactionsResponseFormats collects the response format entries: the
// JSON call format, the option entries in order, and the deprecated image
// config when no image entry was given. Agent calls take no structured
// output.
func interactionsResponseFormats(format *model.ResponseFormat, options InteractionsOptions, isAgent bool) ([]interactionsResponseFormat, []model.Warning) {
	var entries []interactionsResponseFormat
	var warnings []model.Warning

	if format != nil && format.Type == model.ResponseFormatJSON {
		if isAgent {
			warnings = append(warnings, otherWarning(
				"google.interactions: structured output (responseFormat) is not supported when an agent is set; responseFormat will be ignored.",
			))
		} else {
			entries = append(entries, interactionsResponseFormat{
				Type:     InteractionsFormatText,
				MimeType: "application/json",
				Schema:   format.Schema,
			})
		}
	}

	for _, entry := range options.ResponseFormat {
		entries = append(entries, interactionsResponseFormatOf(entry))
	}

	if isAgent || options.ImageConfig == nil {
		return entries, warnings
	}

	hasImageEntry := false
	for _, entry := range entries {
		hasImageEntry = hasImageEntry || entry.Type == InteractionsFormatImage
	}

	if hasImageEntry {
		warnings = append(warnings, otherWarning(
			"google.interactions: providerOptions.google.imageConfig is deprecated and was ignored because "+
				"providerOptions.google.responseFormat already supplies an image entry. Use responseFormat exclusively.",
		))

		return entries, warnings
	}

	warnings = append(warnings, otherWarning(
		"google.interactions: providerOptions.google.imageConfig is deprecated. "+
			"Use providerOptions.google.responseFormat with a { type: \"image\", ... } entry instead.",
	))

	entries = append(entries, interactionsResponseFormat{
		Type:        InteractionsFormatImage,
		MimeType:    "image/png",
		AspectRatio: options.ImageConfig.AspectRatio,
		ImageSize:   options.ImageConfig.ImageSize,
	})

	return entries, warnings
}

// interactionsResponseFormatOf keeps the settings that apply to the
// entry's modality.
func interactionsResponseFormatOf(entry InteractionsResponseFormat) interactionsResponseFormat {
	converted := interactionsResponseFormat{
		Type: entry.Type,
	}

	switch entry.Type {
	case InteractionsFormatText:
		converted.MimeType = entry.MimeType
		converted.Schema = entry.Schema

	case InteractionsFormatImage:
		converted.MimeType = entry.MimeType
		converted.AspectRatio = entry.AspectRatio
		converted.ImageSize = entry.ImageSize

	case InteractionsFormatAudio:
		converted.MimeType = entry.MimeType

	case InteractionsFormatVideo:
		converted.AspectRatio = entry.AspectRatio
		converted.Resolution = entry.Resolution
		converted.Duration = entry.Duration
		converted.Delivery = entry.Delivery
		converted.GCSURI = entry.GCSURI
	}

	return converted
}

// interactionsGenerationConfigOf returns the generation config of a model
// call, or nil when nothing is set.
func interactionsGenerationConfigOf(options model.CallOptions, googleOptions InteractionsOptions, toolChoice *interactionsToolChoice) *interactionsGenerationConfig {
	config := interactionsGenerationConfig{
		Temperature:       options.Temperature,
		TopP:              options.TopP,
		TopK:              options.TopK,
		Seed:              options.Seed,
		MaxOutputTokens:   options.MaxOutputTokens,
		ThinkingLevel:     googleOptions.ThinkingLevel,
		ThinkingSummaries: googleOptions.ThinkingSummaries,
		ToolChoice:        toolChoice,
	}

	if len(options.StopSequences) > 0 {
		config.StopSequences = options.StopSequences
	}

	if config.isEmpty() {
		return nil
	}

	return &config
}

// droppedAgentFieldsWarning lists the generation settings an agent call
// does not accept.
func droppedAgentFieldsWarning(options model.CallOptions, googleOptions InteractionsOptions) []model.Warning {
	candidates := []struct {
		name string
		set  bool
	}{
		{
			name: "temperature",
			set:  options.Temperature != nil,
		},
		{
			name: "topP",
			set:  options.TopP != nil,
		},
		{
			name: "topK",
			set:  options.TopK != nil,
		},
		{
			name: "frequencyPenalty",
			set:  options.FrequencyPenalty != nil,
		},
		{
			name: "presencePenalty",
			set:  options.PresencePenalty != nil,
		},
		{
			name: "seed",
			set:  options.Seed != nil,
		},
		{
			name: "stopSequences",
			set:  len(options.StopSequences) > 0,
		},
		{
			name: "maxOutputTokens",
			set:  options.MaxOutputTokens != nil,
		},
		{
			name: "thinkingLevel",
			set:  googleOptions.ThinkingLevel != "",
		},
		{
			name: "thinkingSummaries",
			set:  googleOptions.ThinkingSummaries != "",
		},
		{
			name: "imageConfig",
			set:  googleOptions.ImageConfig != nil,
		},
	}

	var dropped []string

	for _, candidate := range candidates {
		if candidate.set {
			dropped = append(dropped, candidate.name)
		}
	}

	if len(dropped) == 0 {
		return nil
	}

	verb := "are"
	if len(dropped) == 1 {
		verb = "is"
	}

	return []model.Warning{
		otherWarning(fmt.Sprintf(
			"google.interactions: %s %s not supported when an agent is set; use providerOptions.google.agentConfig instead. Dropped from the request body.",
			strings.Join(dropped, ", "),
			verb,
		)),
	}
}

func interactionsAgentConfigOf(config *InteractionsAgentConfig) *interactionsAgentConfig {
	if config == nil {
		return nil
	}

	if config.Type == InteractionsAgentDynamic {
		return &interactionsAgentConfig{
			Type: InteractionsAgentDynamic,
		}
	}

	return &interactionsAgentConfig{
		Type:                  config.Type,
		ThinkingSummaries:     config.ThinkingSummaries,
		Visualization:         config.Visualization,
		CollaborativePlanning: config.CollaborativePlanning,
	}
}
