package grokbuild

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/aarock1234/ai/internal/schema"
	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/acp"
	"github.com/aarock1234/ai/pkg/model"
)

// questionRequestMethod is the JSON-RPC method of Grok Build's question
// requests.
const questionRequestMethod = "_x.ai/ask_user_question"

// questionNativeName is Grok Build's name of its question tool.
const questionNativeName = "ask_user_question"

// providerOptionsKey is the provider options namespace of tool results
// that request a native question outcome.
const providerOptionsKey = "grok-build"

// Positional ID prefixes of the askUserQuestions input.
const (
	questionIDPrefix = "question-"
	optionIDPrefix   = "option-"
)

// Native question outcomes.
const (
	outcomeAccepted      = "accepted"
	outcomeCancelled     = "cancelled"
	outcomeChatAboutThis = "chat_about_this"
	outcomeSkipInterview = "skip_interview"
)

// errJSONToolResult reports an askUserQuestions result without JSON output.
var errJSONToolResult = errors.New("grok build askUserQuestions requires a json tool result")

// questionsOutputValidator validates askUserQuestions results.
var questionsOutputValidator = sync.OnceValues(func() (*schema.Validator, error) {
	return schema.Compile(json.RawMessage(harness.QuestionsToolOutputSchema))
})

// askUserQuestions maps Grok Build's question requests to the common
// askUserQuestions tool.
func askUserQuestions() *acp.AskUserQuestionsSettings {
	return &acp.AskUserQuestionsSettings{
		RequestMethod:        questionRequestMethod,
		FromNativeRequest:    fromNativeQuestionRequest,
		ToNativeResponse:     toNativeQuestionResponse,
		MatchesNativeRequest: matchesNativeQuestionRequest,
	}
}

// questionRequest is a Grok Build question request.
type questionRequest struct {
	SessionID  *string           `json:"sessionId"`
	ToolCallID *string           `json:"toolCallId"`
	Questions  *[]nativeQuestion `json:"questions"`
	Mode       *string           `json:"mode"`
}

// nativeQuestion is one question of a request.
type nativeQuestion struct {
	Question    *string                 `json:"question"`
	Options     *[]nativeQuestionOption `json:"options"`
	MultiSelect *bool                   `json:"multi_select,omitempty"`
}

// nativeQuestionOption is one choice of a question.
type nativeQuestionOption struct {
	Label       *string `json:"label"`
	Description *string `json:"description"`
	Preview     *string `json:"preview,omitempty"`
}

// decodeQuestionRequest decodes a request, reporting false when it lacks a
// required field or has an unknown mode.
func decodeQuestionRequest(raw json.RawMessage) (questionRequest, bool) {
	var request questionRequest
	if err := json.Unmarshal(raw, &request); err != nil {
		return questionRequest{}, false
	}

	if request.SessionID == nil || request.ToolCallID == nil || request.Questions == nil || request.Mode == nil {
		return questionRequest{}, false
	}

	if *request.Mode != "default" && *request.Mode != "plan" {
		return questionRequest{}, false
	}

	for _, question := range *request.Questions {
		if !question.isValid() {
			return questionRequest{}, false
		}
	}

	return request, true
}

// isValid reports a question with its text and complete options.
func (q nativeQuestion) isValid() bool {
	if q.Question == nil || q.Options == nil {
		return false
	}

	return !slices.ContainsFunc(*q.Options, func(option nativeQuestionOption) bool {
		return option.Label == nil || option.Description == nil
	})
}

// fromNativeQuestionRequest converts a request into a client-executed
// askUserQuestions tool call. Malformed requests stay unhandled.
func fromNativeQuestionRequest(native acp.NativeQuestionRequest) (*harness.ToolCallPart, error) {
	request, ok := decodeQuestionRequest(native.NativeRequest)
	if !ok {
		return nil, nil
	}

	input, err := json.Marshal(questionsInput(request))
	if err != nil {
		return nil, fmt.Errorf("encode grok build askUserQuestions input: %w", err)
	}

	return &harness.ToolCallPart{
		ToolCall: model.ToolCall{
			ToolCallID: *request.ToolCallID,
			ToolName:   string(harness.ToolAskUserQuestions),
			Input:      string(input),
		},
		NativeName: questionNativeName,
	}, nil
}

// questionsInput converts a request into the askUserQuestions input, with
// positional question and option IDs.
func questionsInput(request questionRequest) harness.QuestionsToolInput {
	questions := make([]harness.Question, 0, len(*request.Questions))

	for questionIndex, native := range *request.Questions {
		options := make([]harness.QuestionOption, 0, len(*native.Options))

		for optionIndex, option := range *native.Options {
			converted := harness.QuestionOption{
				ID:          positionalID(optionIDPrefix, optionIndex),
				Label:       *option.Label,
				Description: *option.Description,
			}

			if option.Preview != nil {
				converted.Preview = *option.Preview
			}

			options = append(options, converted)
		}

		question := harness.Question{
			ID:       positionalID(questionIDPrefix, questionIndex),
			Question: *native.Question,
			Options:  options,
			AllowFreeForm: harness.QuestionFreeForm{
				Allowed: true,
			},
		}

		if native.MultiSelect != nil {
			question.AllowMultiple = *native.MultiSelect
		}

		questions = append(questions, question)
	}

	return harness.QuestionsToolInput{
		AllowPartialAnswers: true,
		Questions:           questions,
	}
}

// questionResponse is Grok Build's response to a question request.
type questionResponse struct {
	Outcome        string                        `json:"outcome"`
	Answers        map[string][]string           `json:"answers,omitzero"`
	Annotations    map[string]questionAnnotation `json:"annotations,omitempty"`
	PartialAnswers map[string]string             `json:"partial_answers,omitzero"`
}

// questionAnnotation annotates the answer to one question.
type questionAnnotation struct {
	Preview string `json:"preview,omitempty"`
	Notes   string `json:"notes,omitempty"`
}

// nativeAnswers are the answers keyed by question text.
type nativeAnswers struct {
	answers     map[string][]string
	annotations map[string]questionAnnotation
}

// toNativeQuestionResponse converts the askUserQuestions result into Grok
// Build's response. A grok-build provider option outcome of
// chat_about_this or skip_interview returns the answers as partial answers
// with that outcome.
func toNativeQuestionResponse(response acp.NativeQuestionResponse) (any, error) {
	request, ok := decodeQuestionRequest(response.NativeRequest)
	if !ok {
		return nil, errors.New("grok build question request is malformed")
	}

	output, err := parseQuestionsOutput(response.ToolResult.Output)
	if err != nil {
		return nil, err
	}

	if output.Action == harness.QuestionsCancelled {
		return questionResponse{
			Outcome: outcomeCancelled,
		}, nil
	}

	answers := nativeAnswers{
		answers:     map[string][]string{},
		annotations: map[string]questionAnnotation{},
	}

	if output.Action != harness.QuestionsDeclined {
		answers = toNativeAnswers(request, output)
	}

	requestedOutcome := requestedOutcome(response.ToolResult.ProviderOptions)
	if requestedOutcome == outcomeChatAboutThis || requestedOutcome == outcomeSkipInterview {
		return questionResponse{
			Outcome:        requestedOutcome,
			PartialAnswers: partialAnswers(answers.answers),
		}, nil
	}

	if output.Action == harness.QuestionsDeclined {
		return questionResponse{
			Outcome:        outcomeSkipInterview,
			PartialAnswers: map[string]string{},
		}, nil
	}

	return questionResponse{
		Outcome:     outcomeAccepted,
		Answers:     answers.answers,
		Annotations: answers.annotations,
	}, nil
}

// parseQuestionsOutput validates and decodes a JSON askUserQuestions
// result.
func parseQuestionsOutput(output model.ToolResultOutput) (harness.QuestionsToolOutput, error) {
	var value json.RawMessage

	switch typed := output.(type) {
	case model.JSONOutput:
		value = typed.Value
	case model.ErrorJSONOutput:
		value = typed.Value
	default:
		return harness.QuestionsToolOutput{}, errJSONToolResult
	}

	validator, err := questionsOutputValidator()
	if err != nil {
		return harness.QuestionsToolOutput{}, fmt.Errorf("compile askUserQuestions output schema: %w", err)
	}

	if err := validator.Validate(value); err != nil {
		return harness.QuestionsToolOutput{}, fmt.Errorf("grok build askUserQuestions result: %w", err)
	}

	var parsed harness.QuestionsToolOutput
	if err := json.Unmarshal(value, &parsed); err != nil {
		return harness.QuestionsToolOutput{}, fmt.Errorf("decode grok build askUserQuestions result: %w", err)
	}

	return parsed, nil
}

// toNativeAnswers converts the answers into selected option labels keyed by
// question text. A free-form answer selects "Other" and becomes the
// question's notes.
func toNativeAnswers(request questionRequest, output harness.QuestionsToolOutput) nativeAnswers {
	converted := nativeAnswers{
		answers:     map[string][]string{},
		annotations: map[string]questionAnnotation{},
	}

	for questionIndex, question := range *request.Questions {
		answer, ok := output.Answers[positionalID(questionIDPrefix, questionIndex)]
		if !ok {
			continue
		}

		labels := []string{}

		for _, optionID := range answer.OptionIDs {
			optionIndex, ok := positionalIndex(optionIDPrefix, optionID)
			if !ok || optionIndex >= len(*question.Options) {
				continue
			}

			labels = append(labels, *(*question.Options)[optionIndex].Label)
		}

		if answer.Freeform != "" {
			labels = append(labels, "Other")
			converted.annotations[*question.Question] = questionAnnotation{
				Notes: answer.Freeform,
			}
		}

		converted.answers[*question.Question] = labels
	}

	return converted
}

// requestedOutcome returns the outcome the grok-build provider options
// request, or empty.
func requestedOutcome(options model.ProviderOptions) string {
	values, ok, err := model.ProviderValue[map[string]any](options, providerOptionsKey)
	if !ok || err != nil {
		return ""
	}

	outcome, _ := values["outcome"].(string)

	return outcome
}

// partialAnswers joins each question's labels.
func partialAnswers(answers map[string][]string) map[string]string {
	joined := make(map[string]string, len(answers))

	for question, labels := range answers {
		joined[question] = strings.Join(labels, ", ")
	}

	return joined
}

// matchesNativeQuestionRequest reports a re-sent request with the same
// questions and mode.
func matchesNativeQuestionRequest(match acp.NativeQuestionMatch) bool {
	previous, ok := decodeQuestionRequest(match.PreviousNativeRequest)
	if !ok {
		return false
	}

	current, ok := decodeQuestionRequest(match.NativeRequest)
	if !ok {
		return false
	}

	previousFingerprint, err := questionFingerprint(previous)
	if err != nil {
		return false
	}

	currentFingerprint, err := questionFingerprint(current)
	if err != nil {
		return false
	}

	return previousFingerprint == currentFingerprint
}

// questionFingerprint encodes the questions and mode of a request.
func questionFingerprint(request questionRequest) (string, error) {
	encoded, err := json.Marshal(struct {
		Questions []nativeQuestion `json:"questions"`
		Mode      string           `json:"mode"`
	}{
		Questions: *request.Questions,
		Mode:      *request.Mode,
	})
	if err != nil {
		return "", err
	}

	return string(encoded), nil
}

// positionalID returns the one-based ID of an index.
func positionalID(prefix string, index int) string {
	return prefix + strconv.Itoa(index+1)
}

// positionalIndex returns the zero-based index of a positional ID.
func positionalIndex(prefix string, id string) (int, bool) {
	position, ok := strings.CutPrefix(id, prefix)
	if !ok {
		return 0, false
	}

	number, err := strconv.ParseFloat(strings.TrimSpace(position), 64)
	if err != nil || number != math.Trunc(number) || number < 1 || number > math.MaxInt32 {
		return 0, false
	}

	return int(number) - 1, true
}
