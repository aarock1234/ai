package devtools

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// operationStreamText is the operation ID of StreamText calls.
const operationStreamText = "ai.streamText"

// TelemetryOptions configure NewTelemetry.
type TelemetryOptions struct {
	// RunID groups every call into one run. Reuse it to continue an
	// interrupted interaction; the steps of later calls are numbered after
	// the recorded ones. Empty records each call as its own run.
	RunID string

	// Store records the calls. Nil opens the store of the working directory.
	Store *Store

	// Logger receives recording failures. Nil uses slog.Default.
	Logger *slog.Logger
}

// Telemetry is an ai.Telemetry and ai.TextTelemetry integration that
// records GenerateText, StreamText and agent calls. Calls made inside a
// tool's Execute function with the context it receives are recorded as
// child runs of the step that called the tool. Embedding, reranking,
// speech, transcription, translation and decision calls are not recorded.
//
// Go adaptation: upstream also records generateObject and streamObject,
// which this SDK expresses as GenerateText and StreamText outputs. Parent
// runs are found through the tool's context instead of a shared variable,
// so parallel tool calls link correctly. Telemetry methods cannot return
// errors, so recording failures are logged.
type Telemetry struct {
	ai.NopTelemetry

	runID  string
	store  *Store
	logger *slog.Logger

	mu    sync.Mutex
	calls map[string]*callState
}

var (
	_ ai.Telemetry     = (*Telemetry)(nil)
	_ ai.TextTelemetry = (*Telemetry)(nil)
)

// callState tracks one GenerateText or StreamText call.
type callState struct {
	runID            string
	operationType    stepType
	settings         ai.CallSettings
	stepNumberOffset int

	// steps are the running steps by step number.
	steps map[int]stepState
}

// stepState tracks one running step.
type stepState struct {
	id        string
	startedAt time.Time
}

// parentCallKey is the context key of the call whose tool is executing.
type parentCallKey struct{}

// parentCall identifies the call whose tool is executing. Owner tells
// integrations apart.
type parentCall struct {
	owner  *Telemetry
	callID string
}

// NewTelemetry returns a DevTools telemetry integration. Pass it in the
// Integrations of a call's TelemetryOptions.
func NewTelemetry(options TelemetryOptions) (*Telemetry, error) {
	if err := requireDevelopment("NewTelemetry"); err != nil {
		return nil, err
	}

	store, err := storeOrDefault(options.Store)
	if err != nil {
		return nil, err
	}

	return &Telemetry{
		runID:  options.RunID,
		store:  store,
		logger: loggerOrDefault(options.Logger),
		calls:  map[string]*callState{},
	}, nil
}

// OnGenerateTextStart records the run of a call.
func (t *Telemetry) OnGenerateTextStart(ctx context.Context, event ai.TelemetryEvent[ai.GenerateTextStartEvent]) {
	start := event.Event
	parent := t.parentStep(ctx)

	state := &callState{
		runID:         newRunID(time.Now()),
		operationType: operationType(start.OperationID),
		settings:      start.CallSettings,
		steps:         map[int]stepState{},
	}

	if t.runID != "" {
		state.runID = t.runID
		state.stepNumberOffset = t.store.stepCount(t.runID)
	}

	t.mu.Lock()
	t.calls[start.CallID] = state
	t.mu.Unlock()

	newRun := run{
		ID:        state.runID,
		StartedAt: wirejson.Timestamp(time.Now()),
	}

	if parent != nil {
		newRun.ParentRunID = new(parent.runID)
		newRun.ParentStepID = new(parent.stepID)
	}

	if event.FunctionID != "" {
		newRun.FunctionID = new(event.FunctionID)
	}

	if _, err := t.store.createRun(ctx, newRun); err != nil {
		t.logFailure(ctx, "recording run", err)
	}
}

// OnGenerateTextStepStart records a step before its model call.
func (t *Telemetry) OnGenerateTextStepStart(ctx context.Context, event ai.TelemetryEvent[ai.GenerateTextStepStartEvent]) {
	start := event.Event

	t.mu.Lock()
	state, ok := t.calls[start.CallID]
	if !ok {
		t.mu.Unlock()

		return
	}

	current := stepState{
		id:        newStepID(),
		startedAt: time.Now(),
	}
	state.steps[start.StepNumber] = current
	t.mu.Unlock()

	newStep, err := t.newStep(state, start, current)
	if err != nil {
		t.logFailure(ctx, "serializing step input", err)

		return
	}

	t.store.trackActiveStep(current.id, func() stepResult {
		return abortedStepResult(current.startedAt)
	})

	if err := t.store.createStep(ctx, newStep); err != nil {
		t.logFailure(ctx, "recording step", err)
	}
}

// OnGenerateTextStepEnd records the result of a step.
func (t *Telemetry) OnGenerateTextStepEnd(ctx context.Context, event ai.TelemetryEvent[ai.StepResult]) {
	result := event.Event

	current, ok := t.finishStep(result.CallID, result.StepNumber)
	if !ok {
		return
	}

	output, err := serialize(newStepOutput(result))
	if err != nil {
		t.logFailure(ctx, "serializing step output", err)

		return
	}

	usage, err := serialize(newUsageRecord(result.Usage))
	if err != nil {
		t.logFailure(ctx, "serializing step usage", err)

		return
	}

	err = t.store.updateStepResult(ctx, current.id, stepResult{
		Duration:    time.Since(current.startedAt),
		Output:      &output,
		Usage:       &usage,
		RawRequest:  rawJSON(result.Request.Body),
		RawResponse: rawJSON(result.Response.Body),
	})
	if err != nil {
		t.logFailure(ctx, "recording step result", err)
	}
}

// OnGenerateTextEnd forgets a completed call.
func (t *Telemetry) OnGenerateTextEnd(_ context.Context, event ai.TelemetryEvent[ai.GenerateTextEndEvent]) {
	t.mu.Lock()
	defer t.mu.Unlock()

	delete(t.calls, event.Event.CallID)
}

// OnGenerateTextAbort records the running steps of an aborted stream as aborted.
//
// Go adaptation: upstream records aborted steps from its SIGINT and SIGTERM
// handlers; see Store.AbortActiveSteps.
func (t *Telemetry) OnGenerateTextAbort(ctx context.Context, event ai.TelemetryEvent[ai.StreamTextAbortEvent]) {
	t.failRunningSteps(ctx, event.Event.CallID, abortedError)
}

// OnError records the running steps of a failed call with the error.
func (t *Telemetry) OnError(ctx context.Context, event ai.TelemetryEvent[ai.TelemetryErrorEvent]) {
	if event.Event.CallID == "" {
		return
	}

	message := "unknown error"
	if event.Event.Error != nil {
		message = event.Event.Error.Error()
	}

	t.failRunningSteps(ctx, event.Event.CallID, message)
}

// ExecuteTool runs a tool with a context that identifies the call, so that
// calls made inside the tool become child runs of its step.
func (t *Telemetry) ExecuteTool(ctx context.Context, event ai.TelemetryEvent[ai.ToolExecutionStartEvent], execute func(ctx context.Context) error) error {
	toolCtx := context.WithValue(ctx, parentCallKey{}, parentCall{
		owner:  t,
		callID: event.Event.CallID,
	})

	return execute(toolCtx)
}

// parentStepInfo identifies the step whose tool started a call.
type parentStepInfo struct {
	runID  string
	stepID string
}

// parentStep returns the step whose tool is executing in ctx: the latest
// running step of the parent call.
func (t *Telemetry) parentStep(ctx context.Context) *parentStepInfo {
	parent, ok := ctx.Value(parentCallKey{}).(parentCall)
	if !ok || parent.owner != t {
		return nil
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	state, ok := t.calls[parent.callID]
	if !ok {
		return nil
	}

	latestStepNumber := -1
	var latestStep stepState

	for stepNumber, candidate := range state.steps {
		if stepNumber > latestStepNumber {
			latestStepNumber = stepNumber
			latestStep = candidate
		}
	}

	if latestStepNumber < 0 {
		return nil
	}

	return &parentStepInfo{
		runID:  state.runID,
		stepID: latestStep.id,
	}
}

// finishStep removes a running step and returns it.
func (t *Telemetry) finishStep(callID string, stepNumber int) (stepState, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()

	state, ok := t.calls[callID]
	if !ok {
		return stepState{}, false
	}

	current, ok := state.steps[stepNumber]
	if !ok {
		return stepState{}, false
	}

	delete(state.steps, stepNumber)
	t.store.untrackActiveStep(current.id)

	return current, true
}

// failRunningSteps records every running step of a call with the error
// message and forgets the call.
func (t *Telemetry) failRunningSteps(ctx context.Context, callID string, message string) {
	t.mu.Lock()
	state, ok := t.calls[callID]
	delete(t.calls, callID)
	t.mu.Unlock()

	if !ok {
		return
	}

	for _, current := range state.steps {
		t.store.untrackActiveStep(current.id)

		err := t.store.updateStepResult(ctx, current.id, stepResult{
			Duration: time.Since(current.startedAt),
			Error:    new(message),
		})
		if err != nil {
			t.logFailure(ctx, "recording step error", err)
		}
	}
}

func (t *Telemetry) newStep(state *callState, start ai.GenerateTextStepStartEvent, current stepState) (step, error) {
	prompt := start.PromptMessages
	if prompt == nil {
		prompt = start.Messages
	}

	input, err := serialize(stepInput{
		Prompt:           recordJSON.Messages(prompt),
		Tools:            describeTools(start.Tools),
		ToolChoice:       toolChoiceRecord(start.ToolChoice),
		MaxOutputTokens:  state.settings.MaxOutputTokens,
		Temperature:      state.settings.Temperature,
		TopP:             state.settings.TopP,
		TopK:             state.settings.TopK,
		PresencePenalty:  state.settings.PresencePenalty,
		FrequencyPenalty: state.settings.FrequencyPenalty,
		Seed:             state.settings.Seed,
	})
	if err != nil {
		return step{}, err
	}

	providerOptions, err := serializeProviderOptions(start.ProviderOptions)
	if err != nil {
		return step{}, err
	}

	return step{
		ID:              current.id,
		RunID:           state.runID,
		StepNumber:      state.stepNumberOffset + start.StepNumber + 1,
		Type:            state.operationType,
		ModelID:         start.ModelID,
		Provider:        optionalString(start.Provider),
		StartedAt:       wirejson.Timestamp(current.startedAt),
		Input:           input,
		ProviderOptions: providerOptions,
	}, nil
}

func (t *Telemetry) logFailure(ctx context.Context, action string, err error) {
	t.logger.WarnContext(ctx, "devtools telemetry failed "+action, slog.Any("error", err))
}

// stepInput is the recorded input of a step.
type stepInput struct {
	Prompt           []wirejson.Message   `json:"prompt"`
	Tools            []toolDescription    `json:"tools,omitempty"`
	ToolChoice       *wirejson.ToolChoice `json:"toolChoice,omitempty"`
	MaxOutputTokens  *int                 `json:"maxOutputTokens,omitempty"`
	Temperature      *float64             `json:"temperature,omitempty"`
	TopP             *float64             `json:"topP,omitempty"`
	TopK             *int                 `json:"topK,omitempty"`
	PresencePenalty  *float64             `json:"presencePenalty,omitempty"`
	FrequencyPenalty *float64             `json:"frequencyPenalty,omitempty"`
	Seed             *int                 `json:"seed,omitempty"`
}

// toolDescription is the recorded declaration of a tool.
type toolDescription struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// stepOutput is the recorded output of a step.
type stepOutput struct {
	Content      []contentRecord           `json:"content"`
	FinishReason model.UnifiedFinishReason `json:"finishReason"`
	Response     stepOutputResponse        `json:"response"`
}

type stepOutputResponse struct {
	ID        string             `json:"id,omitempty"`
	ModelID   string             `json:"modelId,omitempty"`
	Timestamp string             `json:"timestamp,omitempty"`
	Messages  []wirejson.Message `json:"messages"`
}

func newStepOutput(result ai.StepResult) stepOutput {
	response := stepOutputResponse{
		ID:       result.Response.ID,
		ModelID:  result.Response.ModelID,
		Messages: recordJSON.Messages(result.Response.Messages),
	}

	if !result.Response.Timestamp.IsZero() {
		response.Timestamp = wirejson.Timestamp(result.Response.Timestamp)
	}

	return stepOutput{
		Content:      contentRecords(result.Content),
		FinishReason: result.FinishReason,
		Response:     response,
	}
}

func describeTools(tools []ai.Tool) []toolDescription {
	if tools == nil {
		return nil
	}

	descriptions := make([]toolDescription, 0, len(tools))

	for _, tool := range tools {
		description := toolDescription{
			Name:        tool.Name,
			Description: tool.Description,
		}

		if len(tool.InputSchema) > 0 {
			description.Parameters = wirejson.RawValue(tool.InputSchema)
		}

		descriptions = append(descriptions, description)
	}

	return descriptions
}

func operationType(operationID string) stepType {
	if operationID == operationStreamText {
		return stepStream
	}

	return stepGenerate
}

func abortedStepResult(startedAt time.Time) stepResult {
	return stepResult{
		Duration: time.Since(startedAt),
		Error:    new(abortedError),
	}
}

func optionalString(value string) *string {
	if value == "" {
		return nil
	}

	return &value
}

func storeOrDefault(store *Store) (*Store, error) {
	if store != nil {
		return store, nil
	}

	opened, err := OpenStore(StoreOptions{})
	if err != nil {
		return nil, fmt.Errorf("open devtools store: %w", err)
	}

	return opened, nil
}

func loggerOrDefault(logger *slog.Logger) *slog.Logger {
	if logger != nil {
		return logger
	}

	return slog.Default()
}
