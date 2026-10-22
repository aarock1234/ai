package amazonbedrock

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// generateResult converts a Converse response into content, usage, the
// finish reason and provider metadata.
func (m *ChatModel) generateResult(response converseResponse, prepared *preparedRequest) (*model.GenerateResult, error) {
	if response.Output == nil || response.Output.Message == nil {
		return nil, errMissingField("output.message")
	}

	if response.StopReason == nil {
		return nil, errMissingField("stopReason")
	}

	if len(response.Usage) == 0 {
		return nil, errMissingField("usage")
	}

	usage, err := decodeUsage(response.Usage, true)
	if err != nil {
		return nil, err
	}

	converter := &responseConverter{
		model:    m,
		prepared: prepared,
	}

	if prepared.usesJSONInstruction {
		converter.extractor = &jsonObjectExtractor{}
	}

	for _, block := range response.Output.Message.Content {
		if err := converter.block(block); err != nil {
			return nil, err
		}
	}

	metadata := ChatMetadata{
		Trace:                  traceOf(response.Trace),
		PerformanceConfig:      response.PerformanceConfig,
		ServiceTier:            response.ServiceTier,
		Usage:                  cacheUsageOf(usage),
		IsJSONResponseFromTool: converter.jsonFromTool,
		StopSequence:           response.AdditionalModelResponseFields.stopSequence(),
	}

	return &model.GenerateResult{
		Content:          converter.content,
		FinishReason:     finishReason(*response.StopReason, converter.jsonFromTool),
		Usage:            convertUsage(usage, response.Usage),
		ProviderMetadata: providerMetadataOf(metadata),
	}, nil
}

// responseConverter converts the content blocks of one response in order.
type responseConverter struct {
	model     *ChatModel
	prepared  *preparedRequest
	extractor *jsonObjectExtractor

	content      []model.Content
	jsonFromTool bool
}

func (c *responseConverter) block(block responseBlock) error {
	for _, text := range blockTexts(block) {
		if c.extractor != nil {
			text = c.extractor.process(text)
		}

		c.content = append(c.content, model.Text{
			Text: text,
		})
	}

	if block.ReasoningContent != nil {
		c.reasoning(*block.ReasoningContent)
	}

	if block.ToolUse != nil {
		return c.toolUse(*block.ToolUse)
	}

	return nil
}

// blockTexts returns the text of a block, or the texts of its citations
// content.
func blockTexts(block responseBlock) []string {
	if block.Text != nil {
		return []string{
			*block.Text,
		}
	}

	if block.CitationsContent == nil {
		return nil
	}

	var texts []string
	for _, item := range block.CitationsContent.Content {
		if item.Text != nil {
			texts = append(texts, *item.Text)
		}
	}

	return texts
}

func (c *responseConverter) reasoning(content responseReasoningContent) {
	switch {
	case content.ReasoningText != nil:
		reasoning := model.Reasoning{
			Text: content.ReasoningText.Text,
		}

		if signature := content.ReasoningText.Signature; signature != nil && *signature != "" {
			reasoning.ProviderMetadata = providerMetadataOf(ReasoningMetadata{
				Signature: *signature,
			})
		}

		c.content = append(c.content, reasoning)

	case content.RedactedReasoning != nil:
		c.content = append(c.content, model.Reasoning{
			ProviderMetadata: providerMetadataOf(ReasoningMetadata{
				RedactedData: content.RedactedReasoning.Data,
			}),
		})

	case content.RedactedContent != nil:
		c.content = append(c.content, model.Reasoning{
			ProviderMetadata: providerMetadataOf(ReasoningMetadata{
				RedactedContent: *content.RedactedContent,
			}),
		})
	}
}

// toolUse converts a tool call. A call of the JSON response tool becomes
// the response text.
func (c *responseConverter) toolUse(toolUse responseToolUse) error {
	input, err := compactInput(toolUse.Input)
	if err != nil {
		return err
	}

	if c.prepared.usesJSONResponseTool && toolUse.Name == jsonToolName {
		c.jsonFromTool = true
		c.content = append(c.content, model.Text{
			Text: input,
		})

		return nil
	}

	generateID := c.model.provider.generateID

	toolCallID := toolUse.ToolUseID
	if toolCallID == "" {
		toolCallID = generateID()
	}

	toolName := toolUse.Name
	if toolName == "" {
		toolName = "tool-" + generateID()
	}

	c.content = append(c.content, model.ToolCall{
		ToolCallID: normalizeToolCallID(toolCallID, isMistralModel(c.model.modelID)),
		ToolName:   toolName,
		Input:      input,
	})

	return nil
}

// compactInput returns tool input as compact JSON; absent input is an
// empty object.
func compactInput(input json.RawMessage) (string, error) {
	trimmed := bytes.TrimSpace(input)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return "{}", nil
	}

	var compacted bytes.Buffer
	if err := json.Compact(&compacted, trimmed); err != nil {
		return "", fmt.Errorf("%w: tool input: %w", model.ErrInvalidResponse, err)
	}

	return compacted.String(), nil
}
