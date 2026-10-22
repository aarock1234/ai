package ai

import (
	"cmp"
	"context"
	"iter"
	"time"

	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/pkg/model"
)

// LanguageModelStreamPart is one event of a single streamed language model
// call. The call opens with ModelCallStartPart and, unless the provider
// stream fails, ends with ModelCallEndPart. In between come the text,
// reasoning and tool input parts of StreamText, with provider part IDs, and
// complete items: FileContent, ReasoningFileContent, SourceContent,
// CustomContent, ToolCall (parsed against the tools), ToolResult and
// ToolError (provider-executed results and invalid client calls) and
// ToolApprovalRequest. ModelCallResponseMetadataPart, ErrorPart and RawPart
// may appear anywhere.
type LanguageModelStreamPart interface {
	isLanguageModelStreamPart()
}

// ModelCallStartPart opens a model call with the provider's warnings.
type ModelCallStartPart struct {
	Warnings []model.Warning
}

// ModelCallResponseMetadataPart reports the response metadata the provider
// sent. Zero fields were not reported.
type ModelCallResponseMetadataPart struct {
	model.ResponseMetadata
}

// ModelCallEndPart ends a model call. Its finish reason is
// model.FinishError when the response violated a required tool choice; an
// ErrorPart with the ToolChoiceViolationError follows.
type ModelCallEndPart struct {
	FinishReason    model.UnifiedFinishReason
	RawFinishReason string

	Usage            LanguageModelUsage
	ProviderMetadata model.ProviderMetadata
	Performance      LanguageModelCallPerformance
}

// streamRetryBoundary separates a failed attempt of a streamed step from its
// retry. Only StreamText produces it.
type streamRetryBoundary struct {
	warnings []model.Warning
}

func (ModelCallStartPart) isLanguageModelStreamPart()            {}
func (ModelCallResponseMetadataPart) isLanguageModelStreamPart() {}
func (ModelCallEndPart) isLanguageModelStreamPart()              {}
func (streamRetryBoundary) isLanguageModelStreamPart()           {}

func (TextStartPart) isLanguageModelStreamPart()        {}
func (TextDeltaPart) isLanguageModelStreamPart()        {}
func (TextEndPart) isLanguageModelStreamPart()          {}
func (ReasoningStartPart) isLanguageModelStreamPart()   {}
func (ReasoningDeltaPart) isLanguageModelStreamPart()   {}
func (ReasoningEndPart) isLanguageModelStreamPart()     {}
func (ToolInputStartPart) isLanguageModelStreamPart()   {}
func (ToolInputDeltaPart) isLanguageModelStreamPart()   {}
func (ToolInputEndPart) isLanguageModelStreamPart()     {}
func (ErrorPart) isLanguageModelStreamPart()            {}
func (RawPart) isLanguageModelStreamPart()              {}
func (CustomContent) isLanguageModelStreamPart()        {}
func (SourceContent) isLanguageModelStreamPart()        {}
func (FileContent) isLanguageModelStreamPart()          {}
func (ReasoningFileContent) isLanguageModelStreamPart() {}
func (ToolCall) isLanguageModelStreamPart()             {}
func (ToolResult) isLanguageModelStreamPart()           {}
func (ToolError) isLanguageModelStreamPart()            {}
func (ToolApprovalRequest) isLanguageModelStreamPart()  {}
func (ToolApprovalResponse) isLanguageModelStreamPart() {}

// StreamLanguageModelCallOptions configures StreamLanguageModelCall.
type StreamLanguageModelCallOptions struct {
	Model model.LanguageModel

	// CallSettings configure the call. MaxRetries must be nil: the call is
	// not retried.
	CallSettings
	Prompt

	// Tools are the tools the model may call, in declaration order. Their
	// Execute functions are not called.
	Tools []Tool

	// ToolOrder lists tool names to send first; the other tools follow in
	// name order. Nil keeps declaration order.
	ToolOrder []string

	// ToolChoice controls tool selection. Nil selects automatic choice.
	ToolChoice *model.ToolChoice

	// Output selects the response format. Nil requests text.
	Output Output

	ProviderOptions model.ProviderOptions

	// RepairToolCall attempts to repair tool calls that fail to parse.
	RepairToolCall ToolCallRepairFunc

	// RefineToolInput maps tool names to refinements of their validated
	// inputs.
	RefineToolInput map[string]ToolInputRefinement

	// Download resolves file URLs in the prompt. Nil downloads the URLs the
	// model does not support.
	Download DownloadFunction

	// IncludeRawChunks emits RawPart parts with the provider's native
	// events.
	IncludeRawChunks bool

	// CallID correlates the events of the call. Empty generates one.
	CallID string

	// OnStart receives the converted prompt before the model call starts.
	OnStart func(ctx context.Context, event LanguageModelCallPromptEvent)

	// OnLanguageModelCallStart is called immediately before the model call.
	OnLanguageModelCallStart func(ctx context.Context, event LanguageModelCallStartEvent)

	// OnLanguageModelCallEnd is called once the response has finished.
	OnLanguageModelCallEnd func(ctx context.Context, event LanguageModelCallEndEvent)
}

// LanguageModelCallPromptEvent carries the converted prompt of a model call.
type LanguageModelCallPromptEvent struct {
	PromptMessages []model.Message
}

// StreamLanguageModelCall streams a single language model call after
// standardizing the prompt and preparing the tools. Tool calls are parsed
// against the tools but not executed, and there is no multi-step loop; use
// StreamText for that. It is upstream's experimental_streamLanguageModelCall.
//
// Cancellation and deadlines come from ctx.
func StreamLanguageModelCall(ctx context.Context, options StreamLanguageModelCallOptions) (*LanguageModelCallStream, error) {
	call, err := prepareLanguageModelCall(ctx, options)
	if err != nil {
		return nil, err
	}

	return openLanguageModelCall(ctx, call)
}

// prepareLanguageModelCall validates the options and converts the prompt
// and tools into the model call.
func prepareLanguageModelCall(ctx context.Context, options StreamLanguageModelCallOptions) (languageModelCall, error) {
	if err := validateLanguageModelCallOptions(options); err != nil {
		return languageModelCall{}, err
	}

	callOptions, err := prepareCallSettings(options.CallSettings)
	if err != nil {
		return languageModelCall{}, err
	}

	output := options.Output
	if output == nil {
		output = TextOutput()
	}

	responseFormat, err := output.ResponseFormat()
	if err != nil {
		return languageModelCall{}, err
	}

	prompt, err := standardizePrompt(options.Prompt)
	if err != nil {
		return languageModelCall{}, err
	}

	promptMessages, err := convertToModelPrompt(ctx, convertPromptOptions{
		prompt:        prompt,
		supportedURLs: options.Model.SupportedURLs(),
		download:      options.Download,
	})
	if err != nil {
		return languageModelCall{}, err
	}

	callOptions.Prompt = promptMessages
	callOptions.Tools = prepareTools(options.Tools, options.ToolOrder)
	callOptions.ToolChoice = prepareToolChoice(options.ToolChoice)
	callOptions.ResponseFormat = responseFormat
	callOptions.ProviderOptions = options.ProviderOptions
	callOptions.IncludeRawChunks = options.IncludeRawChunks

	if options.OnStart != nil {
		options.OnStart(ctx, LanguageModelCallPromptEvent{
			PromptMessages: promptMessages,
		})
	}

	return languageModelCall{
		start: LanguageModelCallStartEvent{
			CallID:       cmp.Or(options.CallID, id.New("call")),
			Provider:     options.Model.Provider(),
			ModelID:      options.Model.ModelID(),
			System:       options.System,
			Messages:     prompt.messages,
			Tools:        callOptions.Tools,
			CallSettings: options.CallSettings,
		},
		model:          options.Model,
		callOptions:    callOptions,
		tools:          options.Tools,
		repair:         options.RepairToolCall,
		refine:         options.RefineToolInput,
		collectContent: true,
		onStart:        options.OnLanguageModelCallStart,
		onEnd:          options.OnLanguageModelCallEnd,
	}, nil
}

func validateLanguageModelCallOptions(options StreamLanguageModelCallOptions) error {
	if options.Model == nil {
		return &InvalidArgumentError{
			Parameter: "model",
			Message:   "model must not be nil",
		}
	}

	if options.MaxRetries != nil {
		return &InvalidArgumentError{
			Parameter: "maxRetries",
			Value:     *options.MaxRetries,
			Message:   "a single language model call is not retried; maxRetries must be nil",
		}
	}

	return validateTools(options.Tools)
}

// languageModelCall is one streamed model call, ready to open.
type languageModelCall struct {
	// start describes the call; its System and Messages are passed to the
	// tool call repair function.
	start LanguageModelCallStartEvent

	model       model.LanguageModel
	callOptions model.CallOptions

	// tools are the tools declared to the model, used to parse its calls.
	tools  []Tool
	repair ToolCallRepairFunc
	refine map[string]ToolInputRefinement

	// collectContent records the call's content for the end event and for
	// tool choice errors.
	collectContent bool

	telemetry telemetryDispatcher
	onStart   func(ctx context.Context, event LanguageModelCallStartEvent)
	onEnd     func(ctx context.Context, event LanguageModelCallEndEvent)
}

// LanguageModelCallStream is an open streaming language model call. Range
// over Parts once, or call Close to release it without reading.
type LanguageModelCallStream struct {
	Request  model.RequestInfo
	Response model.StreamResponseInfo

	call    languageModelCall
	stream  *model.StreamResult
	started time.Time
}

// openLanguageModelCall notifies the call start and opens the provider
// stream inside the telemetry wrappers.
func openLanguageModelCall(ctx context.Context, call languageModelCall) (*LanguageModelCallStream, error) {
	if call.onStart != nil {
		call.onStart(ctx, call.start)
	}

	call.telemetry.onLanguageModelCallStart(ctx, call.start)

	started := time.Now()

	var stream *model.StreamResult

	err := call.telemetry.executeLanguageModelCall(ctx, call.start, func(ctx context.Context) error {
		opened, err := call.model.Stream(ctx, call.callOptions)
		stream = opened

		return err
	})
	if err != nil {
		if stream != nil {
			// The open failed; a release failure cannot change that.
			_ = stream.Close()
		}

		return nil, err
	}

	return &LanguageModelCallStream{
		Request:  stream.Request,
		Response: stream.Response,
		call:     call,
		stream:   stream,
		started:  started,
	}, nil
}

// Parts returns the parts of the call. The sequence can be ranged over
// once; breaking out of it releases the provider stream. ctx is used for
// tool call repair and generated file downloads.
func (s *LanguageModelCallStream) Parts(ctx context.Context) iter.Seq[LanguageModelStreamPart] {
	return func(yield func(LanguageModelStreamPart) bool) {
		conversion := newModelCallConversion(s.call, s.started)

		for part := range s.stream.Parts() {
			if !conversion.convert(ctx, part, yield) {
				return
			}
		}
	}
}

// Close releases the provider stream. It is safe to call more than once.
func (s *LanguageModelCallStream) Close() error {
	return s.stream.Close()
}

// modelCallConversion converts the provider parts of one model call into
// language model stream parts.
type modelCallConversion struct {
	call    languageModelCall
	started time.Time

	timing outputTiming
	files  *generatedFileResolver

	// toolCalls are the parsed calls, so that approval requests and
	// provider results can reference them.
	toolCalls []ToolCall

	content    callContentRecorder
	rawContent modelContentRecorder

	responseID      string
	responseModelID string
}

func newModelCallConversion(call languageModelCall, started time.Time) *modelCallConversion {
	return &modelCallConversion{
		call:            call,
		started:         started,
		files:           newGeneratedFileResolver(),
		content:         newCallContentRecorder(call.collectContent),
		rawContent:      newModelContentRecorder(),
		responseID:      id.New("aitxt"),
		responseModelID: call.model.ModelID(),
	}
}

// convert converts one provider part. It reports false when the consumer
// stopped.
func (c *modelCallConversion) convert(ctx context.Context, part model.StreamPart, yield func(LanguageModelStreamPart) bool) bool {
	if isOutputStreamPart(part) {
		c.timing.observe(time.Now(), c.started)
	}

	if c.call.collectContent {
		c.rawContent.record(part)
	}

	switch typed := part.(type) {
	case model.StreamStart:
		return yield(ModelCallStartPart{
			Warnings: typed.Warnings,
		})

	case model.TextStart:
		c.content.startText(typed.ID, typed.ProviderMetadata)

		return yield(TextStartPart{
			ID:               typed.ID,
			ProviderMetadata: typed.ProviderMetadata,
		})

	case model.TextDelta:
		c.content.appendText(typed.ID, typed.Delta, typed.ProviderMetadata)

		return yield(TextDeltaPart{
			ID:               typed.ID,
			Text:             typed.Delta,
			ProviderMetadata: typed.ProviderMetadata,
		})

	case model.TextEnd:
		c.content.endText(typed.ID, typed.Citations, typed.ProviderMetadata)

		return yield(TextEndPart{
			ID:               typed.ID,
			Citations:        typed.Citations,
			ProviderMetadata: typed.ProviderMetadata,
		})

	case model.ReasoningStart:
		c.content.startReasoning(typed.ID, typed.ProviderMetadata)

		return yield(ReasoningStartPart{
			ID:               typed.ID,
			ProviderMetadata: typed.ProviderMetadata,
		})

	case model.ReasoningDelta:
		c.content.appendReasoning(typed.ID, typed.Delta, typed.ProviderMetadata)

		return yield(ReasoningDeltaPart{
			ID:               typed.ID,
			Text:             typed.Delta,
			ProviderMetadata: typed.ProviderMetadata,
		})

	case model.ReasoningEnd:
		c.content.endReasoning(typed.ID, typed.ProviderMetadata)

		return yield(ReasoningEndPart{
			ID:               typed.ID,
			ProviderMetadata: typed.ProviderMetadata,
		})

	case model.File:
		return yield(c.convertFile(ctx, typed))

	case model.ReasoningFile:
		return yield(c.convertReasoningFile(ctx, typed))

	case model.ToolInputStart:
		return yield(c.convertToolInputStart(typed))

	case model.ToolInputDelta:
		return yield(ToolInputDeltaPart{
			ID:               typed.ID,
			Delta:            typed.Delta,
			ProviderMetadata: typed.ProviderMetadata,
		})

	case model.ToolInputEnd:
		return yield(ToolInputEndPart{
			ID:               typed.ID,
			ProviderMetadata: typed.ProviderMetadata,
		})

	case model.ToolCall:
		return c.convertToolCall(ctx, typed, yield)

	case model.ToolApprovalRequest:
		return yield(c.convertApprovalRequest(typed))

	case model.ToolResult:
		return yield(c.convertProviderToolResult(typed))

	case model.Source:
		source := SourceContent{
			Source: typed,
		}
		c.content.add(source)

		return yield(source)

	case model.Custom:
		custom := CustomContent{
			Kind:             typed.Kind,
			ProviderMetadata: typed.ProviderMetadata,
		}
		c.content.add(custom)

		return yield(custom)

	case model.ResponseMetadata:
		c.updateResponse(typed)

		return yield(ModelCallResponseMetadataPart{
			ResponseMetadata: typed,
		})

	case model.Finish:
		return c.finish(ctx, typed, yield)

	case model.StreamError:
		return yield(ErrorPart{
			Err: typed.Err,
		})

	case model.Raw:
		return yield(RawPart{
			Value: typed.Value,
		})

	default:
		return true
	}
}

// convertFile resolves a generated file.
func (c *modelCallConversion) convertFile(ctx context.Context, part model.File) LanguageModelStreamPart {
	file, err := c.files.resolve(ctx, part.Data, part.MediaType)
	if err != nil {
		return ErrorPart{
			Err: err,
		}
	}

	content := FileContent{
		File:             file,
		ProviderMetadata: part.ProviderMetadata,
	}
	c.content.add(content)

	return content
}

// convertReasoningFile resolves a generated reasoning file.
func (c *modelCallConversion) convertReasoningFile(ctx context.Context, part model.ReasoningFile) LanguageModelStreamPart {
	file, err := c.files.resolve(ctx, part.Data, part.MediaType)
	if err != nil {
		return ErrorPart{
			Err: err,
		}
	}

	content := ReasoningFileContent{
		File:             file,
		ProviderMetadata: part.ProviderMetadata,
	}
	c.content.add(content)

	return content
}

// convertToolInputStart adds the tool's kind, title and metadata to the
// start of a streamed tool input.
func (c *modelCallConversion) convertToolInputStart(part model.ToolInputStart) ToolInputStartPart {
	tool, found := findTool(c.call.tools, part.ToolName)

	start := ToolInputStartPart{
		ID:               part.ID,
		ToolName:         part.ToolName,
		InputEncoding:    part.InputEncoding,
		ProviderExecuted: part.ProviderExecuted,
		Dynamic:          part.Dynamic || (found && tool.kind() == ToolKindDynamic),
		ProviderMetadata: part.ProviderMetadata,
	}

	if found {
		start.Title = tool.Title
		start.ToolMetadata = tool.Metadata
	}

	return start
}

// convertToolCall parses a tool call. Invalid client calls are followed by
// the ToolError that reports them to the model.
func (c *modelCallConversion) convertToolCall(ctx context.Context, part model.ToolCall, yield func(LanguageModelStreamPart) bool) bool {
	call, err := parseToolCall(ctx, toolCallParseRequest{
		call:     part,
		tools:    c.call.tools,
		repair:   c.call.repair,
		refine:   c.call.refine,
		system:   c.call.start.System,
		messages: c.call.start.Messages,
	})
	if err != nil {
		return yield(ErrorPart{
			Err: err,
		})
	}

	c.toolCalls = append(c.toolCalls, call)
	c.content.add(call)

	if !yield(call) {
		return false
	}

	if !call.Invalid || call.ProviderExecuted {
		return true
	}

	return yield(invalidToolCallError(call))
}

func (c *modelCallConversion) convertApprovalRequest(part model.ToolApprovalRequest) LanguageModelStreamPart {
	call, found := findToolCall(c.toolCalls, part.ToolCallID)
	if !found {
		return ErrorPart{
			Err: &ToolCallNotFoundForApprovalError{
				ToolCallID: part.ToolCallID,
				ApprovalID: part.ApprovalID,
			},
		}
	}

	request := ToolApprovalRequest{
		ApprovalID: part.ApprovalID,
		ToolCall:   call,
	}
	c.content.add(request)

	return request
}

func (c *modelCallConversion) convertProviderToolResult(part model.ToolResult) LanguageModelStreamPart {
	call, found := findToolCall(c.toolCalls, part.ToolCallID)
	if !found {
		call = ToolCall{
			Dynamic: part.Dynamic,
		}
	}

	output := newProviderToolOutput(call, part)
	c.content.add(output)

	return output
}

func (c *modelCallConversion) updateResponse(metadata model.ResponseMetadata) {
	c.responseID = cmp.Or(metadata.ID, c.responseID)
	c.responseModelID = cmp.Or(metadata.ModelID, c.responseModelID)
}

// finish notifies the end of the call and closes it. A violated tool choice
// turns the finish reason into an error and is reported after the end part.
func (c *modelCallConversion) finish(ctx context.Context, part model.Finish, yield func(LanguageModelStreamPart) bool) bool {
	usage := newLanguageModelUsage(part.Usage)
	performance := c.timing.performance(usage, time.Since(c.started))

	c.notifyEnd(ctx, LanguageModelCallEndEvent{
		CallID:           c.call.start.CallID,
		Provider:         c.call.start.Provider,
		ModelID:          c.responseModelID,
		FinishReason:     part.FinishReason.Unified,
		Usage:            usage,
		Content:          c.content.parts,
		ResponseID:       c.responseID,
		ProviderMetadata: part.ProviderMetadata,
		Performance:      performance,
	})

	violated := enforcedToolChoiceViolation(c.call.callOptions.ToolChoice, c.toolCalls)

	finishReason := part.FinishReason.Unified
	if violated {
		finishReason = model.FinishError
	}

	if !yield(ModelCallEndPart{
		FinishReason:     finishReason,
		RawFinishReason:  part.FinishReason.Raw,
		Usage:            usage,
		ProviderMetadata: part.ProviderMetadata,
		Performance:      performance,
	}) {
		return false
	}

	if !violated {
		return true
	}

	return yield(ErrorPart{
		Err: &ToolChoiceViolationError{
			ToolChoice:   *c.call.callOptions.ToolChoice,
			FinishReason: part.FinishReason,
			Provider:     c.call.model.Provider(),
			ModelID:      c.call.model.ModelID(),
			Content:      c.rawContent.content,
		},
	})
}

func (c *modelCallConversion) notifyEnd(ctx context.Context, event LanguageModelCallEndEvent) {
	if c.call.onEnd != nil {
		c.call.onEnd(ctx, event)
	}

	c.call.telemetry.onLanguageModelCallEnd(ctx, event)
}

// isLanguageModelOutputPart reports whether a part carries generated model
// output, for timeouts and for detecting empty streams.
func isLanguageModelOutputPart(part LanguageModelStreamPart) bool {
	switch typed := part.(type) {
	case TextDeltaPart:
		return typed.Text != ""
	case ReasoningDeltaPart:
		return typed.Text != ""
	case ToolInputDeltaPart:
		return typed.Delta != ""
	case FileContent, ReasoningFileContent, ToolCall:
		return true
	default:
		return false
	}
}
