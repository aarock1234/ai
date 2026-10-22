package openai

import (
	"fmt"

	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/internal/toolnames"
	"github.com/aarock1234/ai/pkg/model"
)

// responsesInputOptions configure prompt conversion.
type responsesInputOptions struct {
	namespace         string
	toolNames         toolnames.Mapping
	systemMessageMode ResponsesSystemMessageMode

	// updateUnsupportedReason explains why reasoning effort updates cannot
	// be sent, or is empty when they can.
	updateUnsupportedReason string

	passThroughUnsupportedFiles bool

	// store reports that responses are stored, so earlier items can be
	// referenced by ID.
	store bool

	// isReasoningModel reports that stored assistant messages may only be
	// referenced together with their reasoning.
	isReasoningModel bool

	// hasConversation skips items that the conversation already contains.
	hasConversation bool

	// hasPreviousResponseID skips items that the previous response chain
	// already contains.
	hasPreviousResponseID bool

	hasLocalShellTool  bool
	hasShellTool       bool
	hasApplyPatchTool  bool
	hasComputerTool    bool
	toolSearchToolName string

	customToolNames       map[string]bool
	outputSchemaToolNames map[string]bool

	// explicitMessageItems adds "type": "message" to message items.
	explicitMessageItems bool
}

// messageType returns the type of message items: "message" when message
// items are explicit, otherwise empty so the type is omitted.
func (o responsesInputOptions) messageType() responsesItemType {
	if o.explicitMessageItems {
		return responsesMessage
	}

	return ""
}

// responsesInputConverter converts a prompt into Responses API input items.
type responsesInputConverter struct {
	options  responsesInputOptions
	input    []responsesInputItem
	warnings []model.Warning

	processedApprovalIDs    map[string]bool
	programmaticToolCallIDs map[string]bool

	parallelGroups         map[string]responsesParallelResultGroup
	emittedParallelCalls   map[string]bool
	emittedParallelResults map[string]bool
}

// responsesAssistantState tracks the items of one assistant message.
type responsesAssistantState struct {
	reasoning          map[string]*responsesReasoningInput
	emittedTextItemIDs map[string]bool

	// hasPrecedingReasoning reports that a reasoning part came before. Stored
	// messages of reasoning models are bound to their reasoning item, so a
	// message without it is sent with its content instead of a reference.
	hasPrecedingReasoning bool
}

func newResponsesInputConverter(options responsesInputOptions) *responsesInputConverter {
	return &responsesInputConverter{
		options:                 options,
		input:                   []responsesInputItem{},
		processedApprovalIDs:    make(map[string]bool),
		programmaticToolCallIDs: make(map[string]bool),
		parallelGroups:          make(map[string]responsesParallelResultGroup),
		emittedParallelCalls:    make(map[string]bool),
		emittedParallelResults:  make(map[string]bool),
	}
}

// convert converts the prompt and returns the input with its warnings.
func (c *responsesInputConverter) convert(prompt []model.Message) ([]responsesInputItem, []model.Warning, error) {
	if c.options.hasConversation || c.options.hasPreviousResponseID {
		groups, err := responsesCompleteParallelGroups(prompt, c.options.namespace)
		if err != nil {
			return nil, nil, err
		}

		c.parallelGroups = groups
	}

	for _, message := range prompt {
		if err := c.message(message); err != nil {
			return nil, nil, err
		}
	}

	if !c.options.store {
		c.dropUnencryptedReasoning()
	}

	return c.input, c.warnings, nil
}

func (c *responsesInputConverter) message(message model.Message) error {
	switch message := message.(type) {
	case model.SystemMessage:
		return c.system(message)
	case model.UserMessage:
		return c.user(message)
	case model.AssistantMessage:
		return c.assistant(message)
	case model.ToolMessage:
		return c.tool(message)
	default:
		return fmt.Errorf("%w: unsupported role %q", model.ErrInvalidPrompt, message.Role())
	}
}

// system converts a system message, or the reasoning effort update it
// carries. Updates keep their position so that they apply to the same part
// of the conversation when the history is sent again.
func (c *responsesInputConverter) system(message model.SystemMessage) error {
	options, err := c.systemOptions(message.ProviderOptions)
	if err != nil {
		return err
	}

	if effort := options.ReasoningEffortUpdate; effort != "" {
		reason := c.options.updateUnsupportedReason
		if message.Content != "" {
			reason = "message-level reasoningEffortUpdate requires empty system message content"
		}

		if reason != "" {
			return fmt.Errorf("%w: message-level reasoningEffortUpdate: %s", model.ErrUnsupported, reason)
		}

		c.add(responsesConfigurationUpdateInput{
			Type: responsesConfigurationUpdate,
			Reasoning: responsesConfigurationReasoning{
				Effort: effort,
			},
		})

		return nil
	}

	if c.options.systemMessageMode == ResponsesSystemMessageRemove {
		c.warnOther("system messages are removed for this model")

		return nil
	}

	c.add(responsesSystemMessage{
		Type:    c.options.messageType(),
		Role:    string(c.options.systemMessageMode),
		Content: responsesTextContent(message.Content, options.PromptCacheBreakpoint),
	})

	return nil
}

func (c *responsesInputConverter) systemOptions(providerOptions model.ProviderOptions) (ResponsesSystemMessageOptions, error) {
	options, found, err := model.ProviderValue[ResponsesSystemMessageOptions](providerOptions, c.options.namespace)
	if err != nil {
		return options, err
	}

	if !found && c.options.namespace != Namespace {
		options, _, err = model.ProviderValue[ResponsesSystemMessageOptions](providerOptions, Namespace)
		if err != nil {
			return options, err
		}
	}

	return options, options.validate()
}

// user converts a user message.
func (c *responsesInputConverter) user(message model.UserMessage) error {
	content := make([]responsesContentPart, 0, len(message.Content))

	for index, part := range message.Content {
		switch part := part.(type) {
		case model.TextPart:
			options, err := responsesProviderValue[ResponsesPartOptions](part.ProviderOptions, c.options.namespace)
			if err != nil {
				return err
			}

			content = append(content, responsesInputText{
				Type:                  "input_text",
				Text:                  part.Text,
				PromptCacheBreakpoint: options.PromptCacheBreakpoint,
			})

		case model.FilePart:
			converted, err := c.userFile(part, index)
			if err != nil {
				return err
			}

			content = append(content, converted)
		}
	}

	c.add(responsesUserMessage{
		Type:    c.options.messageType(),
		Role:    "user",
		Content: content,
	})

	return nil
}

// userFile converts a user file. Inline files other than images must be
// PDFs unless pass-through is enabled.
func (c *responsesInputConverter) userFile(part model.FilePart, index int) (responsesContentPart, error) {
	options, err := responsesProviderValue[ResponsesPartOptions](part.ProviderOptions, c.options.namespace)
	if err != nil {
		return nil, err
	}

	isImage := mediatype.TopLevel(part.MediaType) == "image"
	breakpoint := options.PromptCacheBreakpoint

	switch data := part.Data.(type) {
	case model.FileReference:
		fileID, err := c.fileID(data)
		if err != nil {
			return nil, err
		}

		return responsesFileByID(fileID, isImage, options), nil

	case model.FileText:
		return nil, fmt.Errorf("%w: text file parts", model.ErrUnsupported)

	case model.FileURL:
		fileURL, err := responsesURLString(data)
		if err != nil {
			return nil, err
		}

		return responsesFileByURL(fileURL, isImage, options), nil

	case model.FileBytes:
		mediaType, err := mediatype.ResolveFile(part.MediaType, data)
		if err != nil {
			return nil, err
		}

		if isImage {
			return responsesInputImage{
				Type:                  "input_image",
				ImageURL:              dataurl.Encode(mediaType, data.Data),
				Detail:                options.ImageDetail,
				PromptCacheBreakpoint: breakpoint,
			}, nil
		}

		isPDF := mediaType == "application/pdf"
		if !isPDF && !c.options.passThroughUnsupportedFiles {
			return nil, fmt.Errorf("%w: file part media type %s", model.ErrUnsupported, mediaType)
		}

		return responsesInputFile{
			Type:                  "input_file",
			Filename:              responsesPartFilename(part.Filename, index, isPDF),
			FileData:              dataurl.Encode(mediaType, data.Data),
			PromptCacheBreakpoint: breakpoint,
		}, nil

	default:
		return nil, fmt.Errorf("%w: unsupported file data %T", model.ErrInvalidPrompt, part.Data)
	}
}

// assistant converts an assistant message.
func (c *responsesInputConverter) assistant(message model.AssistantMessage) error {
	state := &responsesAssistantState{
		reasoning:          make(map[string]*responsesReasoningInput),
		emittedTextItemIDs: make(map[string]bool),
	}

	for _, part := range message.Content {
		var err error

		switch part := part.(type) {
		case model.TextPart:
			err = c.assistantText(part, state)
		case model.ToolCallPart:
			err = c.assistantToolCall(part)
		case model.ToolResultPart:
			err = c.assistantToolResult(part)
		case model.ReasoningPart:
			err = c.assistantReasoning(part, state)
		case model.CustomPart:
			err = c.assistantCustom(part)
		}

		if err != nil {
			return err
		}
	}

	return nil
}

// assistantText sends stored text as an item reference and other text with
// its content.
func (c *responsesInputConverter) assistantText(part model.TextPart, state *responsesAssistantState) error {
	metadata, err := responsesProviderValue[ResponsesTextMetadata](part.ProviderOptions, c.options.namespace)
	if err != nil {
		return err
	}

	id := metadata.ItemID

	// A conversation already contains its items; sending them again fails
	// with a duplicate item error.
	if c.options.hasConversation && id != "" {
		return nil
	}

	canReference := !c.options.isReasoningModel || state.hasPrecedingReasoning
	if c.options.store && id != "" && canReference {
		if !state.emittedTextItemIDs[id] {
			state.emittedTextItemIDs[id] = true
			c.addReference(id)
		}

		return nil
	}

	c.add(responsesAssistantMessage{
		Type:    c.options.messageType(),
		Role:    "assistant",
		Content: part.Text,
		Phase:   metadata.Phase,
	})

	return nil
}

// assistantReasoning converts reasoning. Stored reasoning is referenced
// once per item; otherwise the summary parts of one item are merged and the
// last encrypted content is kept.
func (c *responsesInputConverter) assistantReasoning(part model.ReasoningPart, state *responsesAssistantState) error {
	state.hasPrecedingReasoning = true

	metadata, err := responsesProviderValue[ResponsesReasoningMetadata](part.ProviderOptions, c.options.namespace)
	if err != nil {
		return err
	}

	reasoningID := metadata.ItemID

	if reasoningID == "" {
		c.unidentifiedReasoning(part, metadata)

		return nil
	}

	if c.options.hasConversation || c.options.hasPreviousResponseID {
		return nil
	}

	existing := state.reasoning[reasoningID]

	if c.options.store {
		if existing == nil {
			c.addReference(reasoningID)
			state.reasoning[reasoningID] = &responsesReasoningInput{
				Type: responsesReasoningItem,
				ID:   reasoningID,
			}
		}

		return nil
	}

	summary := []responsesSummaryText{}
	if part.Text != "" {
		summary = append(summary, responsesSummaryText{
			Type: "summary_text",
			Text: part.Text,
		})
	} else if existing != nil {
		c.warnOther(fmt.Sprintf("Cannot append empty reasoning part to existing reasoning sequence. Skipping reasoning part of item %s.", reasoningID))
	}

	if existing == nil {
		item := &responsesReasoningInput{
			Type:             responsesReasoningItem,
			ID:               reasoningID,
			EncryptedContent: metadata.ReasoningEncryptedContent,
			Summary:          summary,
		}
		state.reasoning[reasoningID] = item
		c.add(item)

		return nil
	}

	existing.Summary = append(existing.Summary, summary...)
	if metadata.ReasoningEncryptedContent != nil {
		existing.EncryptedContent = metadata.ReasoningEncryptedContent
	}

	return nil
}

// unidentifiedReasoning sends reasoning without an item ID when it has
// encrypted content, which the API accepts without stored items.
func (c *responsesInputConverter) unidentifiedReasoning(part model.ReasoningPart, metadata ResponsesReasoningMetadata) {
	if metadata.ReasoningEncryptedContent == nil {
		c.warnOther(fmt.Sprintf("Non-OpenAI reasoning parts are not supported. Skipping reasoning part: %q.", part.Text))

		return
	}

	summary := []responsesSummaryText{}
	if part.Text != "" {
		summary = append(summary, responsesSummaryText{
			Type: "summary_text",
			Text: part.Text,
		})
	}

	c.add(&responsesReasoningInput{
		Type:             responsesReasoningItem,
		EncryptedContent: metadata.ReasoningEncryptedContent,
		Summary:          summary,
	})
}

// assistantCustom converts compaction items. Other custom content is not
// sent.
func (c *responsesInputConverter) assistantCustom(part model.CustomPart) error {
	if part.Kind != "openai.compaction" {
		return nil
	}

	metadata, err := responsesProviderValue[ResponsesCompactionMetadata](part.ProviderOptions, c.options.namespace)
	if err != nil {
		return err
	}

	id := metadata.ItemID
	if id == "" || c.options.hasConversation {
		return nil
	}

	if c.options.store {
		c.addReference(id)

		return nil
	}

	c.add(responsesCompactionInput{
		Type:             responsesCompaction,
		ID:               id,
		EncryptedContent: metadata.EncryptedContent,
	})

	return nil
}

// fileID resolves a file reference for this provider.
func (c *responsesInputConverter) fileID(reference model.FileReference) (string, error) {
	fileID, ok := reference.IDs[c.options.namespace]
	if !ok {
		return "", fmt.Errorf("%w: file reference has no %s file ID", model.ErrInvalidPrompt, c.options.namespace)
	}

	return fileID, nil
}

// isToolSearch reports whether toolName is the call's tool search tool.
func (c *responsesInputConverter) isToolSearch(toolName string) bool {
	return c.options.toolSearchToolName != "" && toolName == c.options.toolSearchToolName
}

// dropUnencryptedReasoning removes reasoning items without encrypted
// content, which the API rejects when responses are not stored.
func (c *responsesInputConverter) dropUnencryptedReasoning() {
	kept := make([]responsesInputItem, 0, len(c.input))

	for _, item := range c.input {
		reasoning, ok := item.(*responsesReasoningInput)
		if ok && reasoning.EncryptedContent == nil {
			continue
		}

		kept = append(kept, item)
	}

	if len(kept) == len(c.input) {
		return
	}

	c.warnOther("Reasoning parts without encrypted content are not supported when store is false. Skipping reasoning parts.")
	c.input = kept
}

func (c *responsesInputConverter) add(item responsesInputItem) {
	c.input = append(c.input, item)
}

func (c *responsesInputConverter) addReference(id string) {
	c.add(responsesItemReference{
		Type: responsesItemReferenceItem,
		ID:   id,
	})
}

func (c *responsesInputConverter) warnOther(message string) {
	c.warnings = append(c.warnings, model.Warning{
		Type:    model.WarningOther,
		Message: message,
	})
}

// responsesTextContent returns text content, as a part when it carries a
// cache breakpoint.
func responsesTextContent(text string, breakpoint *ResponsesPromptCacheBreakpoint) responsesContent {
	if breakpoint == nil {
		return responsesContent{
			Text: text,
		}
	}

	return responsesContent{
		Parts: []responsesContentPart{
			responsesInputText{
				Type:                  "input_text",
				Text:                  text,
				PromptCacheBreakpoint: breakpoint,
			},
		},
	}
}

// responsesFileByID returns an uploaded image or file.
func responsesFileByID(fileID string, isImage bool, options ResponsesPartOptions) responsesContentPart {
	if isImage {
		return responsesInputImage{
			Type:                  "input_image",
			FileID:                fileID,
			Detail:                options.ImageDetail,
			PromptCacheBreakpoint: options.PromptCacheBreakpoint,
		}
	}

	return responsesInputFile{
		Type:                  "input_file",
		FileID:                fileID,
		PromptCacheBreakpoint: options.PromptCacheBreakpoint,
	}
}

// responsesFileByURL returns an image or file that the API downloads.
func responsesFileByURL(fileURL string, isImage bool, options ResponsesPartOptions) responsesContentPart {
	if isImage {
		return responsesInputImage{
			Type:                  "input_image",
			ImageURL:              fileURL,
			Detail:                options.ImageDetail,
			PromptCacheBreakpoint: options.PromptCacheBreakpoint,
		}
	}

	return responsesInputFile{
		Type:                  "input_file",
		FileURL:               fileURL,
		PromptCacheBreakpoint: options.PromptCacheBreakpoint,
	}
}

// responsesPartFilename returns the filename of an inline file, naming
// unnamed files after their position in the message.
func responsesPartFilename(filename string, index int, isPDF bool) string {
	if filename != "" {
		return filename
	}

	if isPDF {
		return fmt.Sprintf("part-%d.pdf", index)
	}

	return fmt.Sprintf("part-%d", index)
}

func responsesURLString(data model.FileURL) (string, error) {
	if data.URL == nil {
		return "", fmt.Errorf("%w: file URL is missing", model.ErrInvalidPrompt)
	}

	return data.URL.String(), nil
}

// responsesProviderValue reads the provider options stored under
// namespace, or the zero value when there are none.
func responsesProviderValue[T any](options model.ProviderOptions, namespace string) (T, error) {
	value, _, err := model.ProviderValue[T](options, namespace)

	return value, err
}

func responsesOptionalString(value string) *string {
	if value == "" {
		return nil
	}

	return &value
}

func responsesFirstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}

	return ""
}
