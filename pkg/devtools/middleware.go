package devtools

import (
	"context"
	"encoding/json"
	"log/slog"
	"sync"
	"time"

	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/ai/middleware"
	"github.com/aarock1234/ai/pkg/model"
)

// MiddlewareOptions configure NewMiddleware.
type MiddlewareOptions struct {
	// Store records the calls. Nil opens the store of the working directory.
	Store *Store

	// Logger receives recording failures. Nil uses slog.Default.
	Logger *slog.Logger
}

// NewMiddleware returns language model middleware that records every call
// of the wrapped model as a step of one run. Create one middleware per
// GenerateText or StreamText call, so that its steps share a run:
//
//	devToolsMiddleware, err := devtools.NewMiddleware(devtools.MiddlewareOptions{})
//	wrapped := middleware.WrapLanguageModel(middleware.WrapLanguageModelOptions{
//		Model:      languageModel,
//		Middleware: []middleware.Language{devToolsMiddleware},
//	})
//
// Streaming calls always request raw chunks to record them, and pass them
// on only when the caller requested them. Recording failures are logged and
// never fail the call.
func NewMiddleware(options MiddlewareOptions) (middleware.Language, error) {
	if err := requireDevelopment("NewMiddleware"); err != nil {
		return middleware.Language{}, err
	}

	store, err := storeOrDefault(options.Store)
	if err != nil {
		return middleware.Language{}, err
	}

	recorder := &middlewareRecorder{
		store:  store,
		logger: loggerOrDefault(options.Logger),
		runID:  newRunID(time.Now()),
	}

	return middleware.Language{
		WrapGenerate: recorder.wrapGenerate,
		WrapStream:   recorder.wrapStream,
	}, nil
}

// middlewareRecorder records the calls of one middleware instance as one
// run.
type middlewareRecorder struct {
	store  *Store
	logger *slog.Logger
	runID  string

	mu         sync.Mutex
	runCreated bool
	stepCount  int
}

func (r *middlewareRecorder) wrapGenerate(ctx context.Context, call middleware.LanguageCall) (*model.GenerateResult, error) {
	startedAt := time.Now()
	stepID := r.startStep(ctx, stepGenerate, call, startedAt)

	result, err := call.DoGenerate(ctx)
	if err != nil {
		r.recordResult(ctx, stepID, stepResult{
			Duration: time.Since(startedAt),
			Error:    new(err.Error()),
		})

		return nil, err
	}

	r.recordResult(ctx, stepID, r.generateResult(ctx, result, startedAt))

	return result, nil
}

func (r *middlewareRecorder) generateResult(ctx context.Context, result *model.GenerateResult, startedAt time.Time) stepResult {
	recorded := stepResult{
		RawRequest:  rawJSON(result.Request.Body),
		RawResponse: rawJSON(result.Response.Body),
	}

	output, err := serialize(generateOutput{
		Content:      recordJSON.Contents(result.Content),
		FinishReason: wirejson.EncodeFinishReason(result.FinishReason),
		Response:     newResponseRecord(result.Response),
	})
	if err != nil {
		r.logFailure(ctx, "serializing step output", err)
	} else {
		recorded.Output = &output
	}

	usage, err := serialize(wirejson.EncodeUsage(result.Usage))
	if err != nil {
		r.logFailure(ctx, "serializing step usage", err)
	} else {
		recorded.Usage = &usage
	}

	recorded.Duration = time.Since(startedAt)

	return recorded
}

func (r *middlewareRecorder) wrapStream(ctx context.Context, call middleware.LanguageCall) (*model.StreamResult, error) {
	startedAt := time.Now()
	stepID := r.startStep(ctx, stepStream, call, startedAt)

	params := call.Params
	params.IncludeRawChunks = true

	source, err := call.Model.Stream(ctx, params)
	if err != nil {
		r.recordResult(ctx, stepID, stepResult{
			Duration: time.Since(startedAt),
			Error:    new(err.Error()),
		})

		return nil, err
	}

	recording := &streamRecording{
		recorder:         r,
		ctx:              ctx,
		stepID:           stepID,
		startedAt:        startedAt,
		source:           source,
		passRawChunks:    call.Params.IncludeRawChunks,
		currentText:      map[string]string{},
		currentReasoning: map[string]string{},
		fullStreamChunks: []model.StreamPart{},
		rawChunks:        []json.RawMessage{},
	}

	r.store.trackActiveStep(stepID, recording.abortedResult)

	return recording.result(), nil
}

// startStep records a step and returns its ID. The run is recorded with
// the first step.
func (r *middlewareRecorder) startStep(ctx context.Context, callType stepType, call middleware.LanguageCall, startedAt time.Time) string {
	stepID := newStepID()
	stepNumber := r.nextStepNumber()

	if err := r.ensureRun(ctx); err != nil {
		r.logFailure(ctx, "recording run", err)
	}

	input, err := serialize(newCallInput(call.Params))
	if err != nil {
		r.logFailure(ctx, "serializing step input", err)

		return stepID
	}

	providerOptions, err := serializeProviderOptions(call.Params.ProviderOptions)
	if err != nil {
		r.logFailure(ctx, "serializing provider options", err)
	}

	err = r.store.createStep(ctx, step{
		ID:              stepID,
		RunID:           r.runID,
		StepNumber:      stepNumber,
		Type:            callType,
		ModelID:         call.Model.ModelID(),
		Provider:        optionalString(call.Model.Provider()),
		StartedAt:       wirejson.Timestamp(startedAt),
		Input:           input,
		ProviderOptions: providerOptions,
	})
	if err != nil {
		r.logFailure(ctx, "recording step", err)
	}

	return stepID
}

func (r *middlewareRecorder) nextStepNumber() int {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.stepCount++

	return r.stepCount
}

// ensureRun records the run once.
func (r *middlewareRecorder) ensureRun(ctx context.Context) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.runCreated {
		return nil
	}

	_, err := r.store.createRun(ctx, run{
		ID:        r.runID,
		StartedAt: wirejson.Timestamp(time.Now()),
	})
	if err != nil {
		return err
	}

	r.runCreated = true

	return nil
}

func (r *middlewareRecorder) recordResult(ctx context.Context, stepID string, result stepResult) {
	r.store.untrackActiveStep(stepID)

	if err := r.store.updateStepResult(ctx, stepID, result); err != nil {
		r.logFailure(ctx, "recording step result", err)
	}
}

func (r *middlewareRecorder) logFailure(ctx context.Context, action string, err error) {
	r.logger.WarnContext(ctx, "devtools middleware failed "+action, slog.Any("error", err))
}

// callInput is the recorded input of a model call.
type callInput struct {
	Prompt           []wirejson.Message        `json:"prompt"`
	Tools            []wirejson.ToolDefinition `json:"tools,omitempty"`
	ToolChoice       *wirejson.ToolChoice      `json:"toolChoice,omitempty"`
	MaxOutputTokens  *int                      `json:"maxOutputTokens,omitempty"`
	Temperature      *float64                  `json:"temperature,omitempty"`
	TopP             *float64                  `json:"topP,omitempty"`
	TopK             *int                      `json:"topK,omitempty"`
	PresencePenalty  *float64                  `json:"presencePenalty,omitempty"`
	FrequencyPenalty *float64                  `json:"frequencyPenalty,omitempty"`
	Seed             *int                      `json:"seed,omitempty"`
	ResponseFormat   *wirejson.ResponseFormat  `json:"responseFormat,omitempty"`
}

func newCallInput(params model.CallOptions) callInput {
	return callInput{
		Prompt:           recordJSON.Messages(params.Prompt),
		Tools:            wirejson.EncodeTools(params.Tools),
		ToolChoice:       toolChoiceRecord(params.ToolChoice),
		MaxOutputTokens:  params.MaxOutputTokens,
		Temperature:      params.Temperature,
		TopP:             params.TopP,
		TopK:             params.TopK,
		PresencePenalty:  params.PresencePenalty,
		FrequencyPenalty: params.FrequencyPenalty,
		Seed:             params.Seed,
		ResponseFormat:   responseFormatRecord(params.ResponseFormat),
	}
}

// generateOutput is the recorded output of a non-streaming model call.
type generateOutput struct {
	Content      []wirejson.Content    `json:"content"`
	FinishReason wirejson.FinishReason `json:"finishReason"`
	Response     responseRecord        `json:"response"`
}
