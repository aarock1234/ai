package anthropic

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/internal/toolnames"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	roleSystem    = "system"
	roleUser      = "user"
	roleAssistant = "assistant"
)

// ToolReferenceOptions are the provider options of a ToolResultCustom item
// that references a tool, which lets a client-side tool search return the
// tools it found. Type must be "tool-reference".
type ToolReferenceOptions struct {
	Type     string `json:"type"`
	ToolName string `json:"toolName"`
}

// promptConversion are the inputs of prompt conversion.
type promptConversion struct {
	prompt        []model.Message
	sendReasoning bool
	warnings      *callWarnings
	cache         *cacheControlValidator
	toolNames     toolnames.Mapping

	// toolsets maps application names of toolset tools, such as the
	// computer toolset, to their API toolset names.
	toolsets map[string]string

	// convertMediaPart is a host's conversion of user file parts, or nil.
	convertMediaPart MediaPartConverter
}

// convertedPrompt is the system prompt and messages of a request, with the
// beta features they require.
type convertedPrompt struct {
	system   []*textBlock
	messages []requestMessage
	betas    *betaSet
}

// promptConverter converts one prompt.
type promptConverter struct {
	promptConversion

	result    *convertedPrompt
	hasSystem bool

	// lastUserMessageIndex is the index of the last message that contains
	// user content, or -1.
	lastUserMessageIndex int
}

// messageGroup is a run of consecutive messages that becomes one Anthropic
// message. Tool messages join the user group.
type messageGroup struct {
	role     string
	messages []model.Message
}

// convertPrompt converts the prompt into Anthropic's system prompt and
// alternating messages.
func convertPrompt(conversion promptConversion) (*convertedPrompt, error) {
	converter := &promptConverter{
		promptConversion: conversion,
		result: &convertedPrompt{
			betas: newBetaSet(),
		},
		lastUserMessageIndex: -1,
	}

	groups, err := groupMessages(conversion.prompt)
	if err != nil {
		return nil, err
	}

	for index, group := range groups {
		isLastGroup := index == len(groups)-1

		switch group.role {
		case roleSystem:
			err = converter.systemGroup(group, index == 0)
		case roleUser:
			err = converter.userGroup(group)
		default:
			err = converter.assistantGroup(group, isLastGroup)
		}

		if err != nil {
			return nil, err
		}
	}

	converter.removeOrphanedCallers()

	return converter.result, nil
}

func groupMessages(prompt []model.Message) ([]messageGroup, error) {
	var groups []messageGroup

	for _, message := range prompt {
		role, err := groupRole(message)
		if err != nil {
			return nil, err
		}

		if len(groups) == 0 || groups[len(groups)-1].role != role {
			groups = append(groups, messageGroup{
				role: role,
			})
		}

		last := &groups[len(groups)-1]
		last.messages = append(last.messages, message)
	}

	return groups, nil
}

func groupRole(message model.Message) (string, error) {
	switch message.(type) {
	case model.SystemMessage:
		return roleSystem, nil
	case model.UserMessage, model.ToolMessage:
		return roleUser, nil
	case model.AssistantMessage:
		return roleAssistant, nil
	default:
		return "", fmt.Errorf("%w: unsupported message type %T", model.ErrInvalidPrompt, message)
	}
}

// convertedSystemMessage is a system message before it is placed in the
// top-level system prompt or in the messages.
type convertedSystemMessage struct {
	content         []requestBlock
	clearAt         ClearAt
	effort          Effort
	toolChangeCount int
}

// systemGroup places initial system messages in the top-level system prompt
// and mid-conversation system messages, which may carry tool changes,
// clearing and effort, in the messages.
func (c *promptConverter) systemGroup(group messageGroup, isFirstGroup bool) error {
	converted := make([]convertedSystemMessage, 0, len(group.messages))
	toolChangeCount := 0
	hasInlineOptions := false

	for _, message := range group.messages {
		system, err := c.systemMessage(message.(model.SystemMessage))
		if err != nil {
			return err
		}

		converted = append(converted, system)
		toolChangeCount += system.toolChangeCount
		hasInlineOptions = hasInlineOptions || system.clearAt != "" || system.effort != ""
	}

	isInitial := !c.hasSystem && toolChangeCount == 0 && !hasInlineOptions
	if isFirstGroup || isInitial {
		c.initialSystem(converted, toolChangeCount)

		return nil
	}

	c.result.betas.add("mid-conversation-system-2026-04-07")

	for _, system := range converted {
		message := requestMessage{
			Role:    roleSystem,
			Content: system.content,
			ClearAt: system.clearAt,
		}

		if system.effort != "" {
			message.OutputConfig = &messageOutputConfig{
				Effort: system.effort,
			}
		}

		c.result.messages = append(c.result.messages, message)

		if system.toolChangeCount > 0 {
			c.result.betas.add("mid-conversation-tool-changes-2026-07-01")
		}

		if system.clearAt != "" {
			c.result.betas.add("mid-conversation-system-clear-at-2026-08-21")
		}

		if system.effort != "" {
			c.result.betas.add("mid-conversation-output-config-2026-07-01")
		}
	}

	return nil
}

func (c *promptConverter) systemMessage(message model.SystemMessage) (convertedSystemMessage, error) {
	options, _, err := model.ProviderValue[SystemMessageOptions](message.ProviderOptions, Namespace)
	if err != nil {
		return convertedSystemMessage{}, err
	}

	if err := options.validate(); err != nil {
		return convertedSystemMessage{}, fmt.Errorf("%w: system message options: %w", model.ErrInvalidPrompt, err)
	}

	content := []requestBlock{}

	// A system message that only carries message-level controls may have
	// empty text; it gets no empty text block.
	controlsOnly := len(options.ToolChanges) > 0 || options.ClearAt != "" || options.Effort != ""
	if message.Content != "" || !controlsOnly {
		content = append(content, &textBlock{
			Type:         blockText,
			Text:         message.Content,
			CacheControl: c.cache.cacheControl(message.ProviderOptions, cacheSystemMessage),
		})
	}

	for _, change := range options.ToolChanges {
		content = append(content, &toolChangeBlock{
			Type: change.Type,
			Tool: toolReference{
				Type: blockToolReference,
				Name: c.toolNames.ToProvider(change.ToolName),
			},
		})
	}

	return convertedSystemMessage{
		content:         content,
		clearAt:         options.ClearAt,
		effort:          options.Effort,
		toolChangeCount: len(options.ToolChanges),
	}, nil
}

// initialSystem sets the top-level system prompt. Tool changes and inline
// options are not supported there; an effort-only message stays in the
// messages.
func (c *promptConverter) initialSystem(converted []convertedSystemMessage, toolChangeCount int) {
	if toolChangeCount > 0 {
		c.warnings.other(
			"tool changes on the initial system message are not supported by Anthropic. " +
				"Configure the initial tool set via the tools option instead. " +
				"The tool changes have been ignored.",
		)
	}

	system := []*textBlock{}

	for _, message := range converted {
		isEffortOnly := len(message.content) == 0 && message.clearAt == "" && message.effort != ""

		switch {
		case isEffortOnly:
			c.result.messages = append(c.result.messages, requestMessage{
				Role:    roleSystem,
				Content: []requestBlock{},
				OutputConfig: &messageOutputConfig{
					Effort: message.effort,
				},
			})
			c.result.betas.add("mid-conversation-output-config-2026-07-01")

		case message.clearAt != "" || message.effort != "":
			c.warnings.other(
				"clearAt and effort on this initial system message are not supported by Anthropic. " +
					"Use a separate effort-only system message with empty content to set effort. " +
					"These options have been ignored.",
			)
		}

		for _, block := range message.content {
			if text, ok := block.(*textBlock); ok {
				system = append(system, text)
			}
		}
	}

	c.result.system = system
	c.hasSystem = true
}

// userGroup combines consecutive user and tool messages into one user
// message.
func (c *promptConverter) userGroup(group messageGroup) error {
	content := []requestBlock{}

	for _, message := range group.messages {
		var blocks []requestBlock
		var err error

		switch typed := message.(type) {
		case model.UserMessage:
			if len(typed.Content) > 0 {
				c.lastUserMessageIndex = len(c.result.messages)
			}

			blocks, err = c.userMessage(typed)
		case model.ToolMessage:
			blocks, err = c.toolMessage(typed)
		}

		if err != nil {
			return err
		}

		content = append(content, blocks...)
	}

	c.result.messages = append(c.result.messages, requestMessage{
		Role:    roleUser,
		Content: content,
	})

	return nil
}

// partCacheControl returns the part's cache breakpoint, falling back to the
// message's breakpoint for the message's last part.
func (c *promptConverter) partCacheControl(
	partOptions model.ProviderOptions,
	partLocation cacheLocation,
	messageOptions model.ProviderOptions,
	messageLocation cacheLocation,
	isLastPart bool,
) *CacheControl {
	if cacheControl := c.cache.cacheControl(partOptions, partLocation); cacheControl != nil {
		return cacheControl
	}

	if !isLastPart {
		return nil
	}

	return c.cache.cacheControl(messageOptions, messageLocation)
}

func (c *promptConverter) userMessage(message model.UserMessage) ([]requestBlock, error) {
	blocks := make([]requestBlock, 0, len(message.Content))

	for index, part := range message.Content {
		isLastPart := index == len(message.Content)-1

		switch typed := part.(type) {
		case model.TextPart:
			cacheControl := c.partCacheControl(typed.ProviderOptions, cacheUserMessagePart, message.ProviderOptions, cacheUserMessage, isLastPart)

			blocks = append(blocks, &textBlock{
				Type:         blockText,
				Text:         typed.Text,
				CacheControl: cacheControl,
			})

		case model.FilePart:
			cacheControl := c.partCacheControl(typed.ProviderOptions, cacheUserMessagePart, message.ProviderOptions, cacheUserMessage, isLastPart)

			block, err := c.userFilePart(typed, cacheControl)
			if err != nil {
				return nil, err
			}

			blocks = append(blocks, block)

		default:
			return nil, fmt.Errorf("%w: unsupported user part %T", model.ErrInvalidPrompt, part)
		}
	}

	return blocks, nil
}

// toolsetName returns the API toolset of a tool, known from the call's
// tools or from the metadata of an earlier response, or an empty string.
func (c *promptConverter) toolsetName(toolName string, options model.ProviderOptions) (string, error) {
	if name, found := c.toolsets[toolName]; found {
		return name, nil
	}

	metadata, _, err := model.ProviderValue[ToolCallMetadata](options, Namespace)
	if err != nil {
		return "", err
	}

	return metadata.ToolsetName, nil
}

// toolInput returns tool call input as a JSON object. The API requires an
// object, so other values are wrapped as {"rawInvalidInput": value}.
func toolInput(input json.RawMessage) json.RawMessage {
	trimmed := bytes.TrimSpace(input)
	if len(trimmed) == 0 {
		return json.RawMessage("{}")
	}

	if jsonobject.IsObject(trimmed) {
		return input
	}

	invalid := json.RawMessage(trimmed)
	if !json.Valid(trimmed) {
		invalid = mustEncodeJSON(string(trimmed))
	}

	return mustEncodeJSON(rawInvalidInput{
		RawInvalidInput: invalid,
	})
}

// rawInvalidInput wraps tool call input that is not a JSON object.
type rawInvalidInput struct {
	RawInvalidInput json.RawMessage `json:"rawInvalidInput"`
}

// trimTrailingAssistantText trims the final prefilled assistant text, since
// the API rejects trailing whitespace there.
func trimTrailingAssistantText(text string, isFinalPart bool) string {
	if !isFinalPart {
		return text
	}

	return strings.TrimSpace(text)
}
