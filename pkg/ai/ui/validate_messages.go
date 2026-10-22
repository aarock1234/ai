package ui

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/aarock1234/ai/internal/optional"

	"github.com/aarock1234/ai/internal/schema"
	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// ValidateMessagesOptions configure ValidateMessages. Metadata, data parts
// and tool inputs are only validated when the corresponding schemas or tools
// are provided.
type ValidateMessagesOptions struct {
	Messages []Message

	// MetadataSchema is the JSON Schema of message metadata.
	MetadataSchema json.RawMessage

	// DataSchemas maps data part names to the JSON Schemas of their data.
	// When set, data parts without a schema are rejected.
	DataSchemas map[string]json.RawMessage

	// Tools validate the inputs of static tool parts against their input
	// schemas, and available outputs against their output schemas when the
	// tools have one. Parts in a final state whose tool is missing, or whose
	// failed or empty input no longer matches, are converted to unavailable
	// dynamic tool parts.
	Tools []ai.Tool

	// RefineToolInput reproduces the refinement of tool inputs, so that
	// inputs recorded before refinement can be checked against the refined
	// inputs.
	RefineToolInput map[string]ai.ToolInputRefinement
}

// ValidateMessages validates UI messages received from a client, such as
// those sent by useChat, and returns them with tool parts normalized. The
// input messages are not modified. Upstream's safeValidateUIMessages is
// covered by the returned error.
func ValidateMessages(ctx context.Context, options ValidateMessagesOptions) ([]Message, error) {
	return validateMessages(ctx, options, false)
}

// validateMessagesForAgent validates messages for an agent, whose tools may
// change between requests: terminal tool parts of tools that are no longer
// registered are converted to unavailable dynamic tool parts.
func validateMessagesForAgent(ctx context.Context, options ValidateMessagesOptions) ([]Message, error) {
	return validateMessages(ctx, options, true)
}

func validateMessages(ctx context.Context, options ValidateMessagesOptions, convertMissingTerminalTools bool) ([]Message, error) {
	if options.Messages == nil {
		return nil, &ai.InvalidArgumentError{
			Parameter: "messages",
			Message:   "messages parameter must be provided",
		}
	}

	if len(options.Messages) == 0 {
		return nil, &ValidationError{
			Field: "messages",
			Cause: errors.New("messages array must not be empty"),
		}
	}

	messages := make([]Message, 0, len(options.Messages))

	for index, message := range options.Messages {
		if err := validateMessageShape(index, message); err != nil {
			return nil, err
		}

		messages = append(messages, message.clone())
	}

	validation := &messageValidation{
		options:       options,
		validators:    make(map[string]*schema.Validator),
		validateTools: options.Tools != nil || convertMissingTerminalTools,
	}

	if err := validation.validate(ctx, messages); err != nil {
		return nil, err
	}

	return messages, nil
}

// messageValidation validates message contents against schemas.
type messageValidation struct {
	options ValidateMessagesOptions

	// validators caches the compiled schemas by name.
	validators map[string]*schema.Validator

	validateTools bool
}

// validate checks the contents of messages, replacing tool parts that are
// converted to dynamic parts.
func (v *messageValidation) validate(ctx context.Context, messages []Message) error {
	for messageIndex, message := range messages {
		field := fmt.Sprintf("messages[%d]", messageIndex)

		if v.options.MetadataSchema != nil {
			if err := v.validateMetadata(field, message); err != nil {
				return err
			}
		}

		for partIndex, part := range message.Parts {
			partField := fmt.Sprintf("%s.parts[%d]", field, partIndex)

			switch typed := part.(type) {
			case DataPart:
				if err := v.validateData(partField, typed); err != nil {
					return err
				}

			case ToolPart:
				validated, err := v.validateToolPart(ctx, partField, typed)
				if err != nil {
					return err
				}

				message.Parts[partIndex] = validated
			}
		}
	}

	return nil
}

func (v *messageValidation) validateMetadata(field string, message Message) error {
	validator, err := v.validator("metadata", v.options.MetadataSchema)
	if err != nil {
		return err
	}

	if err := validator.Validate(jsonOrNull(message.Metadata)); err != nil {
		return &ValidationError{
			Field:    field + ".metadata",
			EntityID: message.ID,
			Cause:    err,
		}
	}

	return nil
}

func (v *messageValidation) validateData(field string, part DataPart) error {
	if v.options.DataSchemas == nil {
		return nil
	}

	dataSchema, found := v.options.DataSchemas[part.Name]
	if !found {
		return &ValidationError{
			Field:      field + ".data",
			EntityName: part.Name,
			EntityID:   part.ID,
			Cause:      fmt.Errorf("no data schema found for data part %s", part.Name),
		}
	}

	validator, err := v.validator(dataTypePrefix+part.Name, dataSchema)
	if err != nil {
		return err
	}

	if err := validator.Validate(jsonOrNull(part.Data)); err != nil {
		return &ValidationError{
			Field:      field + ".data",
			EntityName: part.Name,
			EntityID:   part.ID,
			Cause:      err,
		}
	}

	return nil
}

// validateToolPart validates the input and output of a static tool part and
// returns the part, converted to an unavailable dynamic part when its tool
// is missing or its final input no longer matches.
func (v *messageValidation) validateToolPart(ctx context.Context, field string, part ToolPart) (ToolPart, error) {
	if !v.validateTools || part.isDynamicPart() {
		return part, nil
	}

	tool, found := findTool(v.options.Tools, part.ToolName)
	if !found && isTerminalToolState(part.State) {
		return asUnavailableToolPart(part), nil
	}

	inputField := field + ".input"

	if !found {
		return ToolPart{}, &ValidationError{
			Field:      inputField,
			EntityName: part.ToolName,
			EntityID:   part.ToolCallID,
			Cause:      fmt.Errorf("no tool schema found for tool part %s", part.ToolName),
		}
	}

	convertToDynamic := false

	if hasValidatableInput(part) {
		if inputErr := v.validateToolInput(ctx, tool, part); inputErr != nil {
			if !staysLoadableWithInvalidInput(part) {
				return ToolPart{}, &ValidationError{
					Field:      inputField,
					EntityName: part.ToolName,
					EntityID:   part.ToolCallID,
					Cause:      inputErr,
				}
			}

			convertToDynamic = true
		}
	}

	if err := v.validateToolOutput(field, tool, part); err != nil {
		return ToolPart{}, err
	}

	if convertToDynamic {
		return asUnavailableToolPart(part), nil
	}

	return part, nil
}

// validateToolOutput validates the output of a part whose output is
// available against the tool's output schema, when it has one.
func (v *messageValidation) validateToolOutput(field string, tool ai.Tool, part ToolPart) error {
	if part.State != ToolOutputAvailable || tool.OutputSchema == nil {
		return nil
	}

	validator, err := v.validator("tool output "+tool.Name, tool.OutputSchema)
	if err != nil {
		return err
	}

	if err := validator.Validate(jsonOrNull(part.Output)); err != nil {
		return &ValidationError{
			Field:      field + ".output",
			EntityName: part.ToolName,
			EntityID:   part.ToolCallID,
			Cause:      err,
		}
	}

	return nil
}

// staysLoadableWithInvalidInput reports whether a part whose input no
// longer validates is converted instead of rejected. Failed calls and final
// parts with an empty input must stay loadable, but must not claim to match
// the current tool input type.
func staysLoadableWithInvalidInput(part ToolPart) bool {
	if part.State == ToolOutputError {
		return true
	}

	return part.State == ToolOutputAvailable && isEmptyJSONObject(part.Input)
}

// validateToolInput validates the input recorded before refinement, if any,
// and otherwise the input, against the tool's input schema. A recorded
// pre-refinement input must refine to the part's input.
func (v *messageValidation) validateToolInput(ctx context.Context, tool ai.Tool, part ToolPart) error {
	validator, err := v.validator(toolTypePrefix+tool.Name, tool.InputSchema)
	if err != nil {
		return err
	}

	schemaInput := approvalSchemaInput(part)

	input := part.Input
	if schemaInput != nil {
		input = schemaInput
	}

	if err := validator.Validate(jsonOrNull(input)); err != nil {
		return err
	}

	if schemaInput == nil {
		return nil
	}

	reconstructed := schemaInput
	if refine := v.options.RefineToolInput[tool.Name]; refine != nil {
		refined, err := refine(ctx, schemaInput)
		if err != nil {
			return fmt.Errorf("refine tool input: %w", err)
		}

		reconstructed = refined
	}

	if !equalJSON(reconstructed, part.Input) {
		return errors.New("tool input does not match the input reconstructed from inputSchemaInput")
	}

	return nil
}

// validator returns the compiled schema cached under name.
func (v *messageValidation) validator(name string, rawSchema json.RawMessage) (*schema.Validator, error) {
	if validator, cached := v.validators[name]; cached {
		return validator, nil
	}

	validator, err := schema.Compile(rawSchema)
	if err != nil {
		return nil, fmt.Errorf("compile %s schema: %w", name, err)
	}

	v.validators[name] = validator

	return validator, nil
}

// hasValidatableInput reports whether a tool part's input is validated:
// streaming inputs are incomplete, and failed calls may have no input.
func hasValidatableInput(part ToolPart) bool {
	if part.State == ToolInputStreaming {
		return false
	}

	if part.State != ToolOutputError {
		return true
	}

	return approvalSchemaInput(part) != nil || part.Input != nil
}

func approvalSchemaInput(part ToolPart) json.RawMessage {
	if part.Approval == nil {
		return nil
	}

	return part.Approval.InputSchemaInput
}

func isTerminalToolState(state ToolState) bool {
	return state == ToolOutputAvailable || state == ToolOutputError || state == ToolOutputDenied
}

func asUnavailableToolPart(part ToolPart) ToolPart {
	part.Dynamic = true
	part.Unavailable = true

	return part
}

func isEmptyJSONObject(raw json.RawMessage) bool {
	var object map[string]json.RawMessage
	if err := json.Unmarshal(raw, &object); err != nil {
		return false
	}

	return object != nil && len(object) == 0
}

// jsonOrNull returns raw, or null when it is absent.
func jsonOrNull(raw json.RawMessage) json.RawMessage {
	if raw == nil {
		return json.RawMessage("null")
	}

	return raw
}

// validateMessageShape checks the structure that the upstream message
// schema enforces beyond JSON decoding: roles, states and the fields each
// tool state allows.
func validateMessageShape(index int, message Message) error {
	field := fmt.Sprintf("messages[%d]", index)

	switch message.Role {
	case RoleSystem, RoleUser, RoleAssistant:
	default:
		return &ValidationError{
			Field:    field + ".role",
			EntityID: message.ID,
			Cause:    fmt.Errorf("unsupported role %q", message.Role),
		}
	}

	if message.Role != RoleAssistant && len(message.Parts) == 0 {
		return &ValidationError{
			Field:    field + ".parts",
			EntityID: message.ID,
			Cause:    errors.New("message must contain at least one part"),
		}
	}

	for partIndex, part := range message.Parts {
		if err := validatePartShape(part); err != nil {
			return &ValidationError{
				Field:    fmt.Sprintf("%s.parts[%d]", field, partIndex),
				EntityID: message.ID,
				Cause:    err,
			}
		}
	}

	return nil
}

func validatePartShape(part Part) error {
	switch typed := part.(type) {
	case nil:
		return errors.New("part is nil")
	case TextPart:
		return validatePartState(typed.State)
	case ReasoningPart:
		return validatePartState(typed.State)
	case ToolPart:
		return validateToolPartShape(typed)
	default:
		return nil
	}
}

func validatePartState(state PartState) error {
	switch state {
	case "", PartStreaming, PartDone:
		return nil
	default:
		return fmt.Errorf("unsupported state %q", state)
	}
}

// validateToolPartShape checks that a tool part only has the fields its
// state allows.
func validateToolPartShape(part ToolPart) error {
	switch part.InputEncoding {
	case "", model.ToolInputJSON, model.ToolInputText:
	default:
		return fmt.Errorf("unsupported input encoding %q", part.InputEncoding)
	}

	hasOutput := part.Output != nil
	hasErrorText := part.ErrorText != ""

	switch part.State {
	case ToolInputStreaming, ToolInputAvailable:
		return rejectFields(part.State, hasOutput, hasErrorText, part.Approval != nil)

	case ToolApprovalRequested:
		if err := rejectFields(part.State, hasOutput, hasErrorText, false); err != nil {
			return err
		}

		if part.Approval == nil || part.Approval.Approved != nil || part.Approval.Reason != "" {
			return errors.New("approval-requested state requires an unanswered approval")
		}

		return nil

	case ToolApprovalResponded:
		if err := rejectFields(part.State, hasOutput, hasErrorText, false); err != nil {
			return err
		}

		if part.Approval == nil || part.Approval.Approved == nil {
			return errors.New("approval-responded state requires an answered approval")
		}

		return nil

	case ToolOutputAvailable:
		if err := rejectFields(part.State, false, hasErrorText, false); err != nil {
			return err
		}

		return requireGrantedApproval(part)

	case ToolOutputError:
		if err := rejectFields(part.State, hasOutput, false, false); err != nil {
			return err
		}

		return requireGrantedApproval(part)

	case ToolOutputDenied:
		if err := rejectFields(part.State, hasOutput, hasErrorText, false); err != nil {
			return err
		}

		if part.Approval == nil || optional.Or(part.Approval.Approved, true) {
			return errors.New("output-denied state requires a denied approval")
		}

		return nil

	default:
		return fmt.Errorf("unsupported tool state %q", part.State)
	}
}

// rejectFields reports the first field that is present although the state
// does not allow it.
func rejectFields(state ToolState, hasOutput, hasErrorText, hasApproval bool) error {
	switch {
	case hasOutput:
		return fmt.Errorf("%s state does not allow output", state)
	case hasErrorText:
		return fmt.Errorf("%s state does not allow errorText", state)
	case hasApproval:
		return fmt.Errorf("%s state does not allow approval", state)
	default:
		return nil
	}
}

// requireGrantedApproval accepts parts without an approval and parts whose
// approval was granted.
func requireGrantedApproval(part ToolPart) error {
	if part.Approval == nil {
		return nil
	}

	if part.Approval.Approved == nil || !*part.Approval.Approved {
		return fmt.Errorf("%s state requires a granted approval", part.State)
	}

	return nil
}
