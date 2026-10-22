package ai

import (
	"context"
	"net/http"
	"slices"
	"time"

	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/pkg/model"
)

// decideOperationID and decisionModelCallOperationID identify Decide and
// its model call in events.
const (
	decideOperationID            = "ai.decide"
	decisionModelCallOperationID = "ai.decide.doDecide"
)

// DecideOptions configures Decide.
type DecideOptions struct {
	Model model.DecisionModel

	// State is the shared state every question is decided against:
	// model.TextPart, model.FilePart and model.DecisionJSONPart values. File
	// URLs are downloaded and data URLs decoded before the model is called.
	State []model.DecisionStatePart

	// Questions are the judgments to make. Their IDs key the answers.
	Questions []model.DecisionQuestion

	// MaxOutputTokens limits generated tokens on models that support it. It
	// must be at least 1.
	MaxOutputTokens *int

	// Reasoning selects a reasoning effort on models that support it.
	Reasoning model.ReasoningEffort

	// MaxRetries is the number of retries of the model call. Nil selects 2;
	// zero disables retries.
	MaxRetries *int

	Headers         http.Header
	ProviderOptions model.ProviderOptions

	// Telemetry sends the lifecycle events of the call to the integrations
	// that implement DecisionTelemetry.
	Telemetry TelemetryOptions

	// LogWarnings receives the warnings of the model call. Nil discards
	// them; they are always returned on the result.
	LogWarnings LogWarningsFunc

	// OnStart is called before the model is called, and OnEnd after the
	// answers were validated.
	OnStart func(ctx context.Context, event DecideStartEvent)
	OnEnd   func(ctx context.Context, event DecideEndEvent)
}

// DecideResult is the outcome of Decide.
type DecideResult struct {
	// Answers holds exactly one answer per question, keyed by question ID.
	// Refusals are reported as a *model.DecisionRefusalError instead.
	Answers map[string]model.DecisionAnswer

	Usage DecisionUsage

	Warnings []model.Warning

	// Rounding is the decimal precision the provider declared for
	// probabilities and scores, or nil for full precision.
	Rounding *model.DecisionRounding

	ProviderMetadata model.ProviderMetadata

	// Response always has a timestamp and a model ID.
	Response model.ResponseInfo
}

// DecisionUsage reports the tokens a decision used. Nil counts were not
// reported.
type DecisionUsage struct {
	InputTokens  *int
	OutputTokens *int

	// TotalTokens is nil unless both other counts are known.
	TotalTokens *int
}

// Decide answers typed questions, such as choices, scores and boolean
// judgments, against one shared state with a decision model. Every answer
// is validated: choices must name an option, distributions must be
// complete and sum to one, and scores must match their distribution. The
// decision API is experimental and may change.
//
// Cancellation and deadlines come from ctx.
func Decide(ctx context.Context, options DecideOptions) (*DecideResult, error) {
	if err := options.validate(); err != nil {
		return nil, err
	}

	retry, err := prepareRetries(options.MaxRetries)
	if err != nil {
		return nil, err
	}

	run := decision{
		options:   options,
		retry:     retry,
		callID:    id.New("call"),
		telemetry: newTelemetryDispatcher(options.Telemetry),
	}

	return run.decide(ctx)
}

// validate checks the options and the question types before any I/O.
func (o DecideOptions) validate() error {
	if o.Model == nil {
		return &InvalidArgumentError{
			Parameter: "model",
			Message:   "model is required",
		}
	}

	if o.MaxOutputTokens != nil && *o.MaxOutputTokens < 1 {
		return &InvalidArgumentError{
			Parameter: "maxOutputTokens",
			Value:     *o.MaxOutputTokens,
			Message:   "maxOutputTokens must be >= 1",
		}
	}

	if err := validateDecisionInput(o.State, o.Questions); err != nil {
		return err
	}

	return checkSupportedQuestionTypes(o.Model, o.Questions)
}

// checkSupportedQuestionTypes rejects questions the model cannot answer.
func checkSupportedQuestionTypes(decisionModel model.DecisionModel, questions []model.DecisionQuestion) error {
	supported := decisionModel.SupportedQuestionTypes()

	for _, question := range questions {
		if slices.Contains(supported, question.QuestionType()) {
			continue
		}

		return &model.DecisionUnsupportedQuestionTypeError{
			QuestionID:   question.QuestionID(),
			QuestionType: question.QuestionType(),
			Provider:     decisionModel.Provider(),
			ModelID:      decisionModel.ModelID(),
		}
	}

	return nil
}

// decision is one Decide call.
type decision struct {
	options   DecideOptions
	retry     retryPolicy
	callID    string
	telemetry telemetryDispatcher
}

// decide runs the call and reports a failure to telemetry.
func (d decision) decide(ctx context.Context) (*DecideResult, error) {
	startEvent := d.startEvent()

	if d.options.OnStart != nil {
		d.options.OnStart(ctx, startEvent)
	}

	d.telemetry.onDecideStart(ctx, startEvent)

	result, err := d.run(ctx, startEvent)
	if err != nil {
		return nil, d.telemetry.onOperationError(ctx, d.callID, err)
	}

	return result, nil
}

// run prepares the state, calls the model, validates its answers and
// reports the outcome.
func (d decision) run(ctx context.Context, startEvent DecideStartEvent) (*DecideResult, error) {
	state, err := prepareDecisionState(ctx, d.options.State)
	if err != nil {
		return nil, err
	}

	response, err := d.callAndValidate(ctx, state)
	if err != nil {
		return nil, err
	}

	return d.finish(ctx, startEvent, response), nil
}

// callAndValidate calls the model inside its telemetry events and checks
// that every answer is valid and none is a refusal.
func (d decision) callAndValidate(ctx context.Context, state []model.DecisionStatePart) (*model.DecisionResult, error) {
	callStartEvent := d.modelCallStartEvent(state)
	d.telemetry.onDecisionModelCallStart(ctx, callStartEvent)

	response, err := d.callModel(ctx, state)
	if err != nil {
		return nil, err
	}

	if err := validateDecisionAnswers(d.options.Questions, response.Answers, response.Rounding); err != nil {
		return nil, err
	}

	if err := d.checkRefusals(response.Answers); err != nil {
		return nil, err
	}

	d.telemetry.onDecisionModelCallEnd(ctx, DecisionModelCallEndEvent{
		DecisionModelCallStartEvent: callStartEvent,
		Result:                      *response,
	})

	return response, nil
}

// finish logs the warnings of a valid response, builds the result and
// reports the end of the call.
func (d decision) finish(ctx context.Context, startEvent DecideStartEvent, response *model.DecisionResult) *DecideResult {
	d.options.LogWarnings.log(ctx, WarningsEvent{
		Warnings: response.Warnings,
		Provider: d.options.Model.Provider(),
		ModelID:  d.options.Model.ModelID(),
	})

	result := d.result(response)

	endEvent := DecideEndEvent{
		DecideStartEvent: startEvent,
		Result:           *result,
	}

	if d.options.OnEnd != nil {
		d.options.OnEnd(ctx, endEvent)
	}

	d.telemetry.onDecideEnd(ctx, endEvent)

	return result
}

// callModel calls the model with retries.
func (d decision) callModel(ctx context.Context, state []model.DecisionStatePart) (*model.DecisionResult, error) {
	request := model.DecisionOptions{
		State:           state,
		Questions:       d.options.Questions,
		MaxOutputTokens: d.options.MaxOutputTokens,
		Reasoning:       d.options.Reasoning,
		Headers:         d.options.Headers,
		ProviderOptions: d.options.ProviderOptions,
	}

	response, err := retryCall(ctx, d.retry, func(ctx context.Context) (*model.DecisionResult, error) {
		return d.options.Model.Decide(ctx, request)
	})
	if err != nil {
		return nil, err
	}

	if ctx.Err() != nil {
		return nil, context.Cause(ctx)
	}

	return response, nil
}

// checkRefusals reports refused questions, in question order.
func (d decision) checkRefusals(answers map[string]model.DecisionAnswer) error {
	var refused []string

	for _, question := range d.options.Questions {
		if _, isRefusal := answers[question.QuestionID()].(model.RefusalAnswer); isRefusal {
			refused = append(refused, question.QuestionID())
		}
	}

	if len(refused) == 0 {
		return nil
	}

	return &model.DecisionRefusalError{
		QuestionIDs: refused,
		Provider:    d.options.Model.Provider(),
		ModelID:     d.options.Model.ModelID(),
	}
}

// result converts the model's response, filling in the response timestamp
// and model ID when the provider left them out.
func (d decision) result(response *model.DecisionResult) *DecideResult {
	result := &DecideResult{
		Answers:          response.Answers,
		Usage:            newDecisionUsage(response.Usage),
		Warnings:         response.Warnings,
		Rounding:         response.Rounding,
		ProviderMetadata: response.ProviderMetadata,
		Response:         response.Response,
	}

	if result.Response.Timestamp.IsZero() {
		result.Response.Timestamp = time.Now()
	}

	if result.Response.ModelID == "" {
		result.Response.ModelID = d.options.Model.ModelID()
	}

	return result
}

func newDecisionUsage(usage *model.DecisionUsage) DecisionUsage {
	if usage == nil {
		return DecisionUsage{}
	}

	return DecisionUsage{
		InputTokens:  usage.InputTokens,
		OutputTokens: usage.OutputTokens,
		TotalTokens:  totalDecisionTokens(usage),
	}
}

func totalDecisionTokens(usage *model.DecisionUsage) *int {
	if usage.InputTokens == nil || usage.OutputTokens == nil {
		return nil
	}

	return new(*usage.InputTokens + *usage.OutputTokens)
}
