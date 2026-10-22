package ai

import (
	"fmt"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

// InvalidArgumentError reports an invalid function option. It wraps
// model.ErrInvalidArgument.
type InvalidArgumentError struct {
	Parameter string
	Value     any
	Message   string
}

// Error returns the parameter name and the reason.
func (e *InvalidArgumentError) Error() string {
	return fmt.Sprintf("invalid argument for parameter %s: %s", e.Parameter, e.Message)
}

// Unwrap returns model.ErrInvalidArgument.
func (e *InvalidArgumentError) Unwrap() error {
	return model.ErrInvalidArgument
}

// InvalidPromptError reports a prompt that cannot be standardized. It wraps
// model.ErrInvalidPrompt and the cause, if any.
type InvalidPromptError struct {
	Message string
	Cause   error
}

// Error returns the reason.
func (e *InvalidPromptError) Error() string {
	if e.Cause == nil {
		return "invalid prompt: " + e.Message
	}

	return fmt.Sprintf("invalid prompt: %s: %v", e.Message, e.Cause)
}

// Unwrap returns model.ErrInvalidPrompt and the cause.
func (e *InvalidPromptError) Unwrap() []error {
	return nonNilErrors(model.ErrInvalidPrompt, e.Cause)
}

// InvalidMessageRoleError reports a message whose type is not one of the
// four model message types.
type InvalidMessageRoleError struct {
	Role model.Role
}

// Error returns the offending role.
func (e *InvalidMessageRoleError) Error() string {
	return fmt.Sprintf("invalid message role %q: must be system, user, assistant or tool", e.Role)
}

// Unwrap returns model.ErrInvalidPrompt.
func (e *InvalidMessageRoleError) Unwrap() error {
	return model.ErrInvalidPrompt
}

// InvalidDataContentError reports file content that cannot be decoded, such
// as a malformed data URL.
type InvalidDataContentError struct {
	Content string
	Message string
	Cause   error
}

// Error returns the reason.
func (e *InvalidDataContentError) Error() string {
	if e.Cause == nil {
		return e.Message
	}

	return fmt.Sprintf("%s: %v", e.Message, e.Cause)
}

// Unwrap returns model.ErrInvalidPrompt and the cause.
func (e *InvalidDataContentError) Unwrap() []error {
	return nonNilErrors(model.ErrInvalidPrompt, e.Cause)
}

// MissingToolResultsError reports tool calls in the prompt that have no
// matching tool result before the next user or system message.
type MissingToolResultsError struct {
	ToolCallIDs []string
}

// Error lists the tool calls without results.
func (e *MissingToolResultsError) Error() string {
	if len(e.ToolCallIDs) == 1 {
		return fmt.Sprintf("tool result is missing for tool call %s", e.ToolCallIDs[0])
	}

	return fmt.Sprintf("tool results are missing for tool calls %s", strings.Join(e.ToolCallIDs, ", "))
}

// Unwrap returns model.ErrInvalidPrompt.
func (e *MissingToolResultsError) Unwrap() error {
	return model.ErrInvalidPrompt
}

// NoSuchToolError reports a tool call for a tool that is not available.
type NoSuchToolError struct {
	ToolName string

	// AvailableTools is nil when no tools were available.
	AvailableTools []string
}

// Error names the requested tool and the available ones.
func (e *NoSuchToolError) Error() string {
	if e.AvailableTools == nil {
		return fmt.Sprintf("model tried to call unavailable tool %q: no tools are available", e.ToolName)
	}

	return fmt.Sprintf(
		"model tried to call unavailable tool %q: available tools: %s",
		e.ToolName,
		strings.Join(e.AvailableTools, ", "),
	)
}

// InvalidToolInputError reports tool call arguments that are not valid JSON
// or do not match the tool's input schema.
type InvalidToolInputError struct {
	ToolName  string
	ToolInput string
	Cause     error
}

// Error names the tool and the validation failure.
func (e *InvalidToolInputError) Error() string {
	return fmt.Sprintf("invalid input for tool %s: %v", e.ToolName, e.Cause)
}

// Unwrap returns the validation failure.
func (e *InvalidToolInputError) Unwrap() error {
	return e.Cause
}

// ToolCallRepairError reports a failure of the tool call repair function.
type ToolCallRepairError struct {
	// OriginalError is the *NoSuchToolError or *InvalidToolInputError that
	// triggered the repair.
	OriginalError error

	// Cause is the repair function's error.
	Cause error
}

// Error returns the repair failure.
func (e *ToolCallRepairError) Error() string {
	return fmt.Sprintf("error repairing tool call: %v", e.Cause)
}

// Unwrap returns the repair failure and the original error.
func (e *ToolCallRepairError) Unwrap() []error {
	return nonNilErrors(e.Cause, e.OriginalError)
}

// ToolChoiceViolationError reports a response that does not satisfy a
// required tool choice.
type ToolChoiceViolationError struct {
	// ToolChoice is the required or tool-specific choice that was violated.
	ToolChoice model.ToolChoice

	FinishReason model.FinishReason
	Provider     string
	ModelID      string

	// Content is the model output, which may contain a tool call that the
	// provider returned as text.
	Content []model.Content
}

// Error describes the violated choice.
func (e *ToolChoiceViolationError) Error() string {
	if e.ToolChoice.Type == model.ToolChoiceRequired {
		return "model response did not contain a tool call even though tool choice was required"
	}

	return fmt.Sprintf("model response did not contain a call to the required tool %q", e.ToolChoice.ToolName)
}

// InvalidToolApprovalError reports an approval response whose approval ID has
// no matching approval request in the message history.
type InvalidToolApprovalError struct {
	ApprovalID string
}

// Error names the unknown approval.
func (e *InvalidToolApprovalError) Error() string {
	return fmt.Sprintf(
		"tool approval response references unknown approval ID %q: no matching tool approval request found in message history",
		e.ApprovalID,
	)
}

// InvalidToolApprovalSignatureError reports an approval whose signature does
// not verify.
type InvalidToolApprovalSignatureError struct {
	ApprovalID string
	ToolCallID string
	Reason     string
}

// Error names the approval and the verification failure.
func (e *InvalidToolApprovalSignatureError) Error() string {
	return fmt.Sprintf(
		"tool approval signature verification failed for approval %q (tool call %q): %s",
		e.ApprovalID,
		e.ToolCallID,
		e.Reason,
	)
}

// ToolCallNotFoundForApprovalError reports an approval request that refers
// to an unknown tool call.
type ToolCallNotFoundForApprovalError struct {
	ToolCallID string
	ApprovalID string
}

// Error names the missing tool call.
func (e *ToolCallNotFoundForApprovalError) Error() string {
	return fmt.Sprintf("tool call %q not found for approval request %q", e.ToolCallID, e.ApprovalID)
}

// InvalidStreamPartError reports a stream part that is invalid in its
// position, such as a delta without a start.
type InvalidStreamPartError struct {
	Part    model.StreamPart
	Message string
}

// Error returns the reason.
func (e *InvalidStreamPartError) Error() string {
	return e.Message
}

// StreamProviderError reports an error event from a provider after a model
// response stream has started.
type StreamProviderError struct {
	Message string

	// Type and Code are provider-defined. Numeric codes are formatted as
	// decimal strings.
	Type string
	Code string

	// StatusCode is the HTTP-equivalent status, or zero when unknown.
	StatusCode int

	// IsRetryable reports whether repeating the model call may succeed.
	IsRetryable bool

	// Data is the provider's original error payload.
	Data []byte

	Cause error
}

// Error returns the provider's message.
func (e *StreamProviderError) Error() string {
	return e.Message
}

// Unwrap returns the underlying cause.
func (e *StreamProviderError) Unwrap() error {
	return e.Cause
}

// NoOutputGeneratedError reports that a call finished without any model
// output, for example because every step failed.
type NoOutputGeneratedError struct {
	Cause error
}

// Error returns a fixed message with the cause, if any.
func (e *NoOutputGeneratedError) Error() string {
	if e.Cause == nil {
		return "no output generated"
	}

	return fmt.Sprintf("no output generated: %v", e.Cause)
}

// Unwrap returns the underlying cause.
func (e *NoOutputGeneratedError) Unwrap() error {
	return e.Cause
}

// NoObjectGeneratedError reports that structured output could not be
// produced: the model generated no text, text that is not valid JSON, or JSON
// that does not match the output schema.
type NoObjectGeneratedError struct {
	Message string

	// Text is the text the model generated, if any.
	Text string

	Response     model.ResponseInfo
	Usage        LanguageModelUsage
	FinishReason model.UnifiedFinishReason

	Cause error
}

// Error returns the message with the cause, if any.
func (e *NoObjectGeneratedError) Error() string {
	message := e.Message
	if message == "" {
		message = "no object generated"
	}

	if e.Cause == nil {
		return message
	}

	return fmt.Sprintf("%s: %v", message, e.Cause)
}

// Unwrap returns the parsing or validation failure.
func (e *NoObjectGeneratedError) Unwrap() error {
	return e.Cause
}

// NoImageGeneratedError reports that no image model call produced an image.
type NoImageGeneratedError struct {
	// Calls are the underlying image model calls, including empty ones.
	Calls []GenerateImageCall

	Responses []model.ResponseInfo
	Cause     error
}

// Error returns a fixed message with the cause, if any.
func (e *NoImageGeneratedError) Error() string {
	if e.Cause == nil {
		return "no image generated"
	}

	return fmt.Sprintf("no image generated: %v", e.Cause)
}

// Unwrap returns model.ErrNoContentGenerated and the cause.
func (e *NoImageGeneratedError) Unwrap() []error {
	return nonNilErrors(model.ErrNoContentGenerated, e.Cause)
}

// NoSpeechGeneratedError reports a speech call that returned no audio.
type NoSpeechGeneratedError struct {
	Responses []model.ResponseInfo
}

// Error returns a fixed message.
func (e *NoSpeechGeneratedError) Error() string {
	return "no speech audio generated"
}

// Unwrap returns model.ErrNoContentGenerated.
func (e *NoSpeechGeneratedError) Unwrap() error {
	return model.ErrNoContentGenerated
}

// NoTranscriptGeneratedError reports a transcription call that returned no
// transcript.
type NoTranscriptGeneratedError struct {
	Responses []model.ResponseInfo
}

// Error returns a fixed message.
func (e *NoTranscriptGeneratedError) Error() string {
	return "no transcript generated"
}

// Unwrap returns model.ErrNoContentGenerated.
func (e *NoTranscriptGeneratedError) Unwrap() error {
	return model.ErrNoContentGenerated
}

// NoVideoGeneratedError reports that no video model call produced a video.
type NoVideoGeneratedError struct {
	// Responses has one entry per model call.
	Responses []VideoResponse
}

// Error returns a fixed message.
func (e *NoVideoGeneratedError) Error() string {
	return "no video generated"
}

// Unwrap returns model.ErrNoContentGenerated.
func (e *NoVideoGeneratedError) Unwrap() error {
	return model.ErrNoContentGenerated
}

// NoSuchProviderError reports a registry lookup for an unregistered
// provider. It wraps model.ErrNoSuchModel.
type NoSuchProviderError struct {
	ProviderID         string
	ModelID            string
	ModelType          model.Kind
	AvailableProviders []string
}

// Error names the provider and the registered ones.
func (e *NoSuchProviderError) Error() string {
	return fmt.Sprintf(
		"no such provider %q (available providers: %s)",
		e.ProviderID,
		strings.Join(e.AvailableProviders, ", "),
	)
}

// Unwrap returns model.ErrNoSuchModel.
func (e *NoSuchProviderError) Unwrap() error {
	return model.ErrNoSuchModel
}

func nonNilErrors(errs ...error) []error {
	present := make([]error, 0, len(errs))
	for _, err := range errs {
		if err != nil {
			present = append(present, err)
		}
	}

	return present
}
