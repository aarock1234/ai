package google

import (
	"encoding/json"
	"fmt"
	"maps"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/model"
)

// realtimeThinkingModelPattern matches Live models that reason in the
// background, such as gemini-3.8-live-extended-thinking. They require a
// thinking level or budget; other Live models reject a thinking config.
var realtimeThinkingModelPattern = regexp.MustCompile(`^gemini-\d+\.\d+-live\b.*thinking`)

// RealtimeOptions are the Google settings of a realtime session, read from
// RealtimeSessionConfig.ProviderOptions["google"]. Other provider option
// namespaces are merged into the setup as top-level members, such as a raw
// "generationConfig".
type RealtimeOptions struct {
	// TranslationConfig configures Gemini Live translation, required by
	// gemini-3.5-live-translate-preview to translate into a target
	// language.
	TranslationConfig *RealtimeTranslationConfig `json:"translationConfig,omitempty"`

	// ThinkingConfig configures Live models with background reasoning. They
	// require a thinking level or budget; without either, the provider sends
	// the low level. Latency-optimized Live models reject it.
	ThinkingConfig *ThinkingConfig `json:"thinkingConfig,omitempty"`

	// DefaultToolBehavior is set on every function declaration. Gemini 3.8
	// Live models default to non-blocking function calls; background
	// reasoning models accept only non-blocking calls.
	DefaultToolBehavior RealtimeToolBehavior `json:"defaultToolBehavior,omitempty"`
}

// RealtimeTranslationConfig configures Gemini Live translation.
type RealtimeTranslationConfig struct {
	// TargetLanguageCode is the BCP-47 code of the output language. The
	// Gemini API default is "en".
	TargetLanguageCode string `json:"targetLanguageCode,omitempty"`

	// EchoTargetLanguage echoes input audio already in the target language
	// instead of producing silence.
	EchoTargetLanguage *bool `json:"echoTargetLanguage,omitempty"`
}

// RealtimeToolBehavior is the calling behavior of a Live function.
type RealtimeToolBehavior string

// Realtime tool behaviors.
const (
	RealtimeToolBlocking    RealtimeToolBehavior = "BLOCKING"
	RealtimeToolNonBlocking RealtimeToolBehavior = "NON_BLOCKING"
)

var realtimeToolBehaviors = []RealtimeToolBehavior{
	RealtimeToolBlocking,
	RealtimeToolNonBlocking,
}

// realtimeSetup is the Live API session setup.
type realtimeSetup struct {
	Model                    string                    `json:"model"`
	GenerationConfig         realtimeGenerationConfig  `json:"generationConfig"`
	SystemInstruction        *systemInstruction        `json:"systemInstruction,omitempty"`
	Tools                    []realtimeToolDeclaration `json:"tools,omitempty"`
	InputAudioTranscription  *emptyObject              `json:"inputAudioTranscription,omitempty"`
	OutputAudioTranscription *emptyObject              `json:"outputAudioTranscription,omitempty"`
}

type realtimeGenerationConfig struct {
	ResponseModalities []string      `json:"responseModalities"`
	SpeechConfig       *speechConfig `json:"speechConfig,omitempty"`
}

type realtimeToolDeclaration struct {
	FunctionDeclarations []realtimeFunctionDeclaration `json:"functionDeclarations"`
}

type realtimeFunctionDeclaration struct {
	Name                 string               `json:"name"`
	Description          string               `json:"description,omitempty"`
	ParametersJSONSchema json.RawMessage      `json:"parametersJsonSchema,omitempty"`
	Behavior             RealtimeToolBehavior `json:"behavior,omitempty"`
}

// buildRealtimeSetup converts a normalized configuration. Provider options
// outside the "google" namespace are merged as top-level members; the
// translation and thinking configs are then added to the generation config.
func buildRealtimeSetup(config *model.RealtimeSessionConfig, modelID string) (json.RawMessage, error) {
	if config == nil {
		config = &model.RealtimeSessionConfig{}
	}

	googleOptions, _, err := model.ProviderValue[RealtimeOptions](config.ProviderOptions, Namespace)
	if err != nil {
		return nil, err
	}

	if err := validateEnum("defaultToolBehavior", googleOptions.DefaultToolBehavior, realtimeToolBehaviors); err != nil {
		return nil, err
	}

	encoded, err := json.Marshal(realtimeSetupOf(*config, googleOptions, modelID))
	if err != nil {
		return nil, fmt.Errorf("encode realtime setup: %w", err)
	}

	setup, err := jsonobject.Parse(encoded)
	if err != nil {
		return nil, fmt.Errorf("parse realtime setup: %w", err)
	}

	defaultGeneration, err := objectMember(setup, "generationConfig")
	if err != nil {
		return nil, err
	}

	thinking, err := realtimeThinkingConfig(googleOptions, config.ProviderOptions, modelID)
	if err != nil {
		return nil, err
	}

	if len(config.ProviderOptions) > 0 {
		setup, err = mergeRealtimeOptions(setup, config.ProviderOptions)
		if err != nil {
			return nil, err
		}

		if googleOptions.TranslationConfig != nil {
			translation, err := json.Marshal(googleOptions.TranslationConfig)
			if err != nil {
				return nil, fmt.Errorf("encode realtime translation config: %w", err)
			}

			setup, err = withGenerationMember(setup, defaultGeneration, "translationConfig", translation)
			if err != nil {
				return nil, err
			}
		}
	}

	if thinking != nil {
		encodedThinking, err := thinking.Encode()
		if err != nil {
			return nil, fmt.Errorf("encode realtime thinking config: %w", err)
		}

		setup, err = withGenerationMember(setup, defaultGeneration, "thinkingConfig", encodedThinking)
		if err != nil {
			return nil, err
		}
	}

	return setup.Encode()
}

func realtimeSetupOf(config model.RealtimeSessionConfig, googleOptions RealtimeOptions, modelID string) realtimeSetup {
	setup := realtimeSetup{
		Model: modelPath(modelID),
		GenerationConfig: realtimeGenerationConfig{
			ResponseModalities: realtimeModalities(config.OutputModalities),
		},
	}

	if config.Voice != "" {
		voice := speechVoiceConfig(config.Voice, nil)
		setup.GenerationConfig.SpeechConfig = &voice
	}

	if config.Instructions != "" {
		setup.SystemInstruction = &systemInstruction{
			Parts: []textPart{
				{
					Text: config.Instructions,
				},
			},
		}
	}

	if len(config.Tools) > 0 {
		setup.Tools = []realtimeToolDeclaration{
			{
				FunctionDeclarations: realtimeFunctionDeclarations(config.Tools, googleOptions.DefaultToolBehavior),
			},
		}
	}

	if config.InputAudioTranscription != nil {
		setup.InputAudioTranscription = &emptyObject{}
	}

	if config.OutputAudioTranscription != nil {
		setup.OutputAudioTranscription = &emptyObject{}
	}

	return setup
}

// realtimeModalities upper-cases the requested modalities. Audio is the
// default.
func realtimeModalities(modalities []model.RealtimeModality) []string {
	if modalities == nil {
		return []string{
			"AUDIO",
		}
	}

	converted := make([]string, len(modalities))
	for index, modality := range modalities {
		converted[index] = strings.ToUpper(string(modality))
	}

	return converted
}

func realtimeFunctionDeclarations(tools []model.RealtimeToolDefinition, behavior RealtimeToolBehavior) []realtimeFunctionDeclaration {
	declarations := make([]realtimeFunctionDeclaration, len(tools))

	for index, tool := range tools {
		declarations[index] = realtimeFunctionDeclaration{
			Name:                 tool.Name,
			Description:          tool.Description,
			ParametersJSONSchema: tool.Parameters,
			Behavior:             behavior,
		}
	}

	return declarations
}

// realtimeThinkingConfig returns the thinking config to send, or nil.
// Background reasoning models receive the low level unless the effective
// config, the typed option or a raw generationConfig.thinkingConfig, sets a
// level or a budget.
func realtimeThinkingConfig(googleOptions RealtimeOptions, providerOptions model.ProviderOptions, modelID string) (jsonobject.Object, error) {
	explicit, err := explicitThinkingConfig(googleOptions, providerOptions)
	if err != nil {
		return nil, err
	}

	setsThinking := hasMemberValue(explicit, "thinkingLevel") || hasMemberValue(explicit, "thinkingBudget")
	if isRealtimeThinkingModel(modelID) && !setsThinking {
		return explicit.With("thinkingLevel", jsonobject.EncodeString(string(ThinkingLevelLow))), nil
	}

	if googleOptions.ThinkingConfig == nil {
		return nil, nil
	}

	return explicit, nil
}

// explicitThinkingConfig returns the typed thinking config, or the raw
// generationConfig.thinkingConfig object of the provider options.
func explicitThinkingConfig(googleOptions RealtimeOptions, providerOptions model.ProviderOptions) (jsonobject.Object, error) {
	if googleOptions.ThinkingConfig != nil {
		encoded, err := json.Marshal(googleOptions.ThinkingConfig)
		if err != nil {
			return nil, fmt.Errorf("encode realtime thinking config: %w", err)
		}

		return jsonobject.Parse(encoded)
	}

	rawGeneration, ok := providerOptions["generationConfig"]
	if !ok {
		return nil, nil
	}

	encoded, err := json.Marshal(rawGeneration)
	if err != nil {
		return nil, fmt.Errorf("%w: encode realtime provider option %q: %w", model.ErrInvalidArgument, "generationConfig", err)
	}

	generation, err := jsonobject.Parse(encoded)
	if err != nil {
		return nil, nil
	}

	thinking, ok := generation.Get("thinkingConfig")
	if !ok {
		return nil, nil
	}

	object, err := jsonobject.Parse(thinking)
	if err != nil {
		return nil, nil
	}

	return object, nil
}

func isRealtimeThinkingModel(modelID string) bool {
	name := strings.ToLower(modelID[strings.LastIndex(modelID, "/")+1:])

	return realtimeThinkingModelPattern.MatchString(name)
}

// mergeRealtimeOptions sets each provider option outside the "google"
// namespace as a top-level member of the setup, in key order.
func mergeRealtimeOptions(setup jsonobject.Object, providerOptions model.ProviderOptions) (jsonobject.Object, error) {
	for _, key := range slices.Sorted(maps.Keys(providerOptions)) {
		if key == Namespace {
			continue
		}

		value, err := json.Marshal(providerOptions[key])
		if err != nil {
			return nil, fmt.Errorf("%w: encode realtime provider option %q: %w", model.ErrInvalidArgument, key, err)
		}

		setup = setup.With(key, value)
	}

	return setup, nil
}

// withGenerationMember sets a member of the setup's generation config. A
// generation config replaced by one that is not an object gives way to the
// default one.
func withGenerationMember(
	setup jsonobject.Object,
	defaultGeneration jsonobject.Object,
	key string,
	value json.RawMessage,
) (jsonobject.Object, error) {
	generation := defaultGeneration

	if current, ok := setup.Get("generationConfig"); ok {
		if parsed, err := jsonobject.Parse(current); err == nil {
			generation = parsed
		}
	}

	encoded, err := generation.With(key, value).Encode()
	if err != nil {
		return nil, fmt.Errorf("encode realtime generation config: %w", err)
	}

	return setup.With("generationConfig", encoded), nil
}

// objectMember returns a member that holds a JSON object.
func objectMember(object jsonobject.Object, key string) (jsonobject.Object, error) {
	value, ok := object.Get(key)
	if !ok {
		return jsonobject.Object{}, nil
	}

	member, err := jsonobject.Parse(value)
	if err != nil {
		return nil, fmt.Errorf("parse %s: %w", key, err)
	}

	return member, nil
}

// hasMemberValue reports whether a member is present and not null.
func hasMemberValue(object jsonobject.Object, key string) bool {
	value, ok := object.Get(key)

	return ok && wirejson.HasValue(value)
}

// Live API client messages.
type realtimeSetupMessage struct {
	Setup json.RawMessage `json:"setup"`
}

type realtimeInputMessage struct {
	RealtimeInput realtimeClientInput `json:"realtimeInput"`
}

type realtimeClientInput struct {
	Audio          *realtimeAudioBlob `json:"audio,omitempty"`
	AudioStreamEnd bool               `json:"audioStreamEnd,omitempty"`
	Text           *string            `json:"text,omitempty"`
}

type realtimeAudioBlob struct {
	Data     string `json:"data"`
	MimeType string `json:"mimeType"`
}

type realtimeToolResponseMessage struct {
	ToolResponse realtimeToolResponse `json:"toolResponse"`
}

type realtimeToolResponse struct {
	FunctionResponses []realtimeFunctionResponse `json:"functionResponses"`
}

type realtimeFunctionResponse struct {
	ID       string          `json:"id"`
	Name     string          `json:"name,omitempty"`
	Response json.RawMessage `json:"response"`
}

// realtimeOutputWrapper carries a function output that is not an object.
type realtimeOutputWrapper struct {
	Output json.RawMessage `json:"output"`
}

// serializeRealtimeClientEvent maps a normalized event to a Live API
// message, or nil when the event has no Live API form.
func serializeRealtimeClientEvent(event model.RealtimeClientEvent, modelID string, inputAudioRate int64) (json.RawMessage, error) {
	switch typed := event.(type) {
	case model.RealtimeSessionUpdate:
		setup, err := buildRealtimeSetup(&typed.Config, modelID)
		if err != nil {
			return nil, err
		}

		return encodeRealtimeMessage(realtimeSetupMessage{
			Setup: setup,
		})

	case model.RealtimeInputAudioAppend:
		return encodeRealtimeMessage(realtimeInputMessage{
			RealtimeInput: realtimeClientInput{
				Audio: &realtimeAudioBlob{
					Data:     typed.Audio,
					MimeType: "audio/pcm;rate=" + strconv.FormatInt(inputAudioRate, 10),
				},
			},
		})

	case model.RealtimeInputAudioCommit:
		return encodeRealtimeMessage(realtimeInputMessage{
			RealtimeInput: realtimeClientInput{
				AudioStreamEnd: true,
			},
		})

	case model.RealtimeConversationItemCreate:
		return serializeRealtimeItem(typed.Item)

	default:
		return nil, nil
	}
}

func serializeRealtimeItem(item model.RealtimeConversationItem) (json.RawMessage, error) {
	switch typed := item.(type) {
	case model.RealtimeTextMessage:
		return encodeRealtimeMessage(realtimeInputMessage{
			RealtimeInput: realtimeClientInput{
				Text: &typed.Text,
			},
		})

	case model.RealtimeFunctionCallOutput:
		response, err := functionResponseStruct(typed.Output)
		if err != nil {
			return nil, err
		}

		return encodeRealtimeMessage(realtimeToolResponseMessage{
			ToolResponse: realtimeToolResponse{
				FunctionResponses: []realtimeFunctionResponse{
					{
						ID:       typed.CallID,
						Name:     typed.Name,
						Response: response,
					},
				},
			},
		})

	default:
		return nil, nil
	}
}

// functionResponseStruct returns the response object of a function output.
// The field is a protobuf Struct: an object passes unchanged, and any other
// JSON value, or text that is not JSON, is wrapped under "output".
func functionResponseStruct(output string) (json.RawMessage, error) {
	value := json.RawMessage(output)
	if !json.Valid(value) {
		value = jsonobject.EncodeString(output)
	} else if jsonobject.IsObject(value) {
		return value, nil
	}

	encoded, err := json.Marshal(realtimeOutputWrapper{
		Output: value,
	})
	if err != nil {
		return nil, fmt.Errorf("encode realtime function response: %w", err)
	}

	return encoded, nil
}

func encodeRealtimeMessage(message any) (json.RawMessage, error) {
	encoded, err := json.Marshal(message)
	if err != nil {
		return nil, fmt.Errorf("encode realtime message: %w", err)
	}

	return encoded, nil
}
