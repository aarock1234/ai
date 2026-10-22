package google

import (
	"errors"
	"fmt"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// ChatOptions are the Google provider options of a language model
// call, read from ProviderOptions["google"].
type ChatOptions struct {
	// ResponseModalities selects the output modalities, such as text and
	// images.
	ResponseModalities []ResponseModality `json:"responseModalities,omitempty"`

	ThinkingConfig *ThinkingConfig `json:"thinkingConfig,omitempty"`

	// CachedContent names cached content used as context, in the form
	// "cachedContents/{cachedContent}".
	CachedContent string `json:"cachedContent,omitempty"`

	// StructuredOutputs sends the JSON response schema to the model. It
	// defaults to true; disable it when the schema uses features that Google
	// does not support.
	StructuredOutputs *bool `json:"structuredOutputs,omitempty"`

	// SafetySettings are unique safety settings for blocking unsafe content.
	SafetySettings []SafetySetting `json:"safetySettings,omitempty"`

	// Threshold applies one blocking threshold to every configurable harm
	// category when SafetySettings is empty.
	Threshold SafetyThreshold `json:"threshold,omitempty"`

	// AudioTimestamp enables timestamp understanding for audio-only files.
	AudioTimestamp *bool `json:"audioTimestamp,omitempty"`

	// Labels are billing labels. They are available on Vertex AI only.
	Labels map[string]string `json:"labels,omitempty"`

	MediaResolution MediaResolution `json:"mediaResolution,omitempty"`

	// ImageConfig configures image generation for Gemini image models.
	ImageConfig *ImageConfig `json:"imageConfig,omitempty"`

	// RetrievalConfig provides location context for Google Maps and Google
	// Search grounding.
	RetrievalConfig *RetrievalConfig `json:"retrievalConfig,omitempty"`

	// StreamFunctionCallArguments streams function call arguments
	// incrementally in streaming calls. It is supported on Vertex AI only
	// (see WithVertexAI) and ignored by the Gemini API.
	StreamFunctionCallArguments *bool `json:"streamFunctionCallArguments,omitempty"`

	// ServiceTier selects the Gemini API service tier. Vertex AI ignores it.
	ServiceTier ServiceTier `json:"serviceTier,omitempty"`

	// ContinuationToken resumes a response that stopped with MAX_TOKENS. Pass
	// the token from the previous response's provider metadata with the
	// original prompt and a higher MaxOutputTokens.
	ContinuationToken string `json:"continuationToken,omitempty"`

	// SharedRequestType and RequestType select Vertex AI pay-as-you-go tiers.
	// They are ignored by the Gemini API.
	SharedRequestType SharedRequestType `json:"sharedRequestType,omitempty"`
	RequestType       RequestType       `json:"requestType,omitempty"`
}

// ResponseModality is an output modality.
type ResponseModality string

// Response modalities.
const (
	ResponseModalityText  ResponseModality = "TEXT"
	ResponseModalityImage ResponseModality = "IMAGE"
	ResponseModalityAudio ResponseModality = "AUDIO"
)

// ThinkingConfig configures model thinking.
type ThinkingConfig struct {
	// ThinkingBudget is the number of thinking tokens for Gemini 2.5 models.
	ThinkingBudget *int `json:"thinkingBudget,omitempty"`

	// IncludeThoughts returns thought summaries as reasoning.
	IncludeThoughts *bool `json:"includeThoughts,omitempty"`

	// ThinkingLevel is the thinking depth for Gemini 3 models.
	ThinkingLevel ThinkingLevel `json:"thinkingLevel,omitempty"`
}

// ThinkingLevel is a Gemini 3 thinking depth.
type ThinkingLevel string

// Thinking levels.
const (
	ThinkingLevelMinimal ThinkingLevel = "minimal"
	ThinkingLevelLow     ThinkingLevel = "low"
	ThinkingLevelMedium  ThinkingLevel = "medium"
	ThinkingLevelHigh    ThinkingLevel = "high"
)

// SafetySetting sets the blocking threshold of one harm category.
type SafetySetting struct {
	Category  HarmCategory    `json:"category"`
	Threshold SafetyThreshold `json:"threshold"`
}

// HarmCategory is a safety category.
type HarmCategory string

// Harm categories.
const (
	HarmCategoryUnspecified      HarmCategory = "HARM_CATEGORY_UNSPECIFIED"
	HarmCategoryHateSpeech       HarmCategory = "HARM_CATEGORY_HATE_SPEECH"
	HarmCategoryDangerousContent HarmCategory = "HARM_CATEGORY_DANGEROUS_CONTENT"
	HarmCategoryHarassment       HarmCategory = "HARM_CATEGORY_HARASSMENT"
	HarmCategorySexuallyExplicit HarmCategory = "HARM_CATEGORY_SEXUALLY_EXPLICIT"
	HarmCategoryCivicIntegrity   HarmCategory = "HARM_CATEGORY_CIVIC_INTEGRITY"
)

// SafetyThreshold is a blocking threshold.
type SafetyThreshold string

// Safety thresholds.
const (
	SafetyThresholdUnspecified    SafetyThreshold = "HARM_BLOCK_THRESHOLD_UNSPECIFIED"
	SafetyThresholdLowAndAbove    SafetyThreshold = "BLOCK_LOW_AND_ABOVE"
	SafetyThresholdMediumAndAbove SafetyThreshold = "BLOCK_MEDIUM_AND_ABOVE"
	SafetyThresholdOnlyHigh       SafetyThreshold = "BLOCK_ONLY_HIGH"
	SafetyThresholdNone           SafetyThreshold = "BLOCK_NONE"
	SafetyThresholdOff            SafetyThreshold = "OFF"
)

// MediaResolution is the resolution used for media inputs.
type MediaResolution string

// Media resolutions.
const (
	MediaResolutionUnspecified MediaResolution = "MEDIA_RESOLUTION_UNSPECIFIED"
	MediaResolutionLow         MediaResolution = "MEDIA_RESOLUTION_LOW"
	MediaResolutionMedium      MediaResolution = "MEDIA_RESOLUTION_MEDIUM"
	MediaResolutionHigh        MediaResolution = "MEDIA_RESOLUTION_HIGH"
)

// ImageConfig configures images generated by Gemini models.
type ImageConfig struct {
	AspectRatio AspectRatio `json:"aspectRatio,omitempty"`
	ImageSize   ImageSize   `json:"imageSize,omitempty"`

	// PersonGeneration, ProminentPeople and ImageOutputOptions are Vertex AI
	// options. The Gemini API rejects them, so they are dropped with a
	// warning.
	PersonGeneration   PersonGeneration    `json:"personGeneration,omitempty"`
	ProminentPeople    ProminentPeople     `json:"prominentPeople,omitempty"`
	ImageOutputOptions *ImageOutputOptions `json:"imageOutputOptions,omitempty"`
}

// AspectRatio is the aspect ratio of a generated image.
type AspectRatio string

// Aspect ratios.
const (
	AspectRatio1x1  AspectRatio = "1:1"
	AspectRatio2x3  AspectRatio = "2:3"
	AspectRatio3x2  AspectRatio = "3:2"
	AspectRatio3x4  AspectRatio = "3:4"
	AspectRatio4x3  AspectRatio = "4:3"
	AspectRatio4x5  AspectRatio = "4:5"
	AspectRatio5x4  AspectRatio = "5:4"
	AspectRatio9x16 AspectRatio = "9:16"
	AspectRatio16x9 AspectRatio = "16:9"
	AspectRatio21x9 AspectRatio = "21:9"
	AspectRatio1x8  AspectRatio = "1:8"
	AspectRatio8x1  AspectRatio = "8:1"
	AspectRatio1x4  AspectRatio = "1:4"
	AspectRatio4x1  AspectRatio = "4:1"
)

// ImageSize is the resolution of a generated image.
type ImageSize string

// Image sizes.
const (
	ImageSize512 ImageSize = "512"
	ImageSize1K  ImageSize = "1K"
	ImageSize2K  ImageSize = "2K"
	ImageSize4K  ImageSize = "4K"
)

// PersonGeneration controls the generation of people in images.
type PersonGeneration string

// Person generation settings.
const (
	PersonGenerationUnspecified PersonGeneration = "PERSON_GENERATION_UNSPECIFIED"
	PersonGenerationAllowAll    PersonGeneration = "ALLOW_ALL"
	PersonGenerationAllowAdult  PersonGeneration = "ALLOW_ADULT"
	PersonGenerationAllowNone   PersonGeneration = "ALLOW_NONE"
)

// ProminentPeople controls the generation of prominent people.
type ProminentPeople string

// Prominent people settings.
const (
	ProminentPeopleUnspecified ProminentPeople = "PROMINENT_PEOPLE_UNSPECIFIED"
	ProminentPeopleAllow       ProminentPeople = "ALLOW_PROMINENT_PEOPLE"
	ProminentPeopleBlock       ProminentPeople = "BLOCK_PROMINENT_PEOPLE"
)

// ImageOutputOptions selects the encoding of generated images.
type ImageOutputOptions struct {
	MimeType           string   `json:"mimeType,omitempty"`
	CompressionQuality *float64 `json:"compressionQuality,omitempty"`
}

// RetrievalConfig configures grounding retrieval.
type RetrievalConfig struct {
	LatLng *LatLng `json:"latLng,omitempty"`
}

// LatLng is a geographic location.
type LatLng struct {
	Latitude  *float64 `json:"latitude"`
	Longitude *float64 `json:"longitude"`
}

// ServiceTier is a Gemini API service tier.
type ServiceTier string

// Service tiers.
const (
	ServiceTierStandard ServiceTier = "standard"
	ServiceTierFlex     ServiceTier = "flex"
	ServiceTierPriority ServiceTier = "priority"
)

// SharedRequestType is a Vertex AI shared pay-as-you-go tier.
type SharedRequestType string

// Shared request types.
const (
	SharedRequestTypePriority SharedRequestType = "priority"
	SharedRequestTypeFlex     SharedRequestType = "flex"
	SharedRequestTypeStandard SharedRequestType = "standard"
)

// RequestType is a Vertex AI request type.
type RequestType string

// RequestTypeShared bypasses Vertex AI provisioned throughput.
const RequestTypeShared RequestType = "shared"

// chatOptions reads and validates the Google options of a call from the
// first of namespaces that holds them.
func chatOptions(options model.ProviderOptions, namespaces []string) (ChatOptions, error) {
	googleOptions, _, err := readProviderValue[ChatOptions](options, namespaces)
	if err != nil {
		return ChatOptions{}, err
	}

	if err := googleOptions.validate(); err != nil {
		return ChatOptions{}, err
	}

	return googleOptions, nil
}

// Allowed values of the validated enums.
var (
	responseModalities = []ResponseModality{
		ResponseModalityText,
		ResponseModalityImage,
		ResponseModalityAudio,
	}

	thinkingLevels = []ThinkingLevel{
		ThinkingLevelMinimal,
		ThinkingLevelLow,
		ThinkingLevelMedium,
		ThinkingLevelHigh,
	}

	harmCategories = []HarmCategory{
		HarmCategoryUnspecified,
		HarmCategoryHateSpeech,
		HarmCategoryDangerousContent,
		HarmCategoryHarassment,
		HarmCategorySexuallyExplicit,
		HarmCategoryCivicIntegrity,
	}

	safetyThresholds = []SafetyThreshold{
		SafetyThresholdUnspecified,
		SafetyThresholdLowAndAbove,
		SafetyThresholdMediumAndAbove,
		SafetyThresholdOnlyHigh,
		SafetyThresholdNone,
		SafetyThresholdOff,
	}

	mediaResolutions = []MediaResolution{
		MediaResolutionUnspecified,
		MediaResolutionLow,
		MediaResolutionMedium,
		MediaResolutionHigh,
	}

	aspectRatios = []AspectRatio{
		AspectRatio1x1,
		AspectRatio2x3,
		AspectRatio3x2,
		AspectRatio3x4,
		AspectRatio4x3,
		AspectRatio4x5,
		AspectRatio5x4,
		AspectRatio9x16,
		AspectRatio16x9,
		AspectRatio21x9,
		AspectRatio1x8,
		AspectRatio8x1,
		AspectRatio1x4,
		AspectRatio4x1,
	}

	imageSizes = []ImageSize{
		ImageSize512,
		ImageSize1K,
		ImageSize2K,
		ImageSize4K,
	}

	personGenerations = []PersonGeneration{
		PersonGenerationUnspecified,
		PersonGenerationAllowAll,
		PersonGenerationAllowAdult,
		PersonGenerationAllowNone,
	}

	prominentPeopleSettings = []ProminentPeople{
		ProminentPeopleUnspecified,
		ProminentPeopleAllow,
		ProminentPeopleBlock,
	}

	imageOutputMimeTypes = []string{
		"image/jpeg",
		"image/png",
	}

	serviceTiers = []ServiceTier{
		ServiceTierStandard,
		ServiceTierFlex,
		ServiceTierPriority,
	}

	sharedRequestTypes = []SharedRequestType{
		SharedRequestTypePriority,
		SharedRequestTypeFlex,
		SharedRequestTypeStandard,
	}

	requestTypes = []RequestType{
		RequestTypeShared,
	}
)

func (o ChatOptions) validate() error {
	for _, modality := range o.ResponseModalities {
		if err := validateEnum("responseModalities", modality, responseModalities); err != nil {
			return err
		}
	}

	if o.ThinkingConfig != nil {
		if err := validateEnum("thinkingConfig.thinkingLevel", o.ThinkingConfig.ThinkingLevel, thinkingLevels); err != nil {
			return err
		}
	}

	for _, setting := range o.SafetySettings {
		if err := setting.validate(); err != nil {
			return err
		}
	}

	if err := o.ImageConfig.validate(); err != nil {
		return err
	}

	if err := o.RetrievalConfig.validate(); err != nil {
		return err
	}

	return errors.Join(
		validateEnum("threshold", o.Threshold, safetyThresholds),
		validateEnum("mediaResolution", o.MediaResolution, mediaResolutions),
		validateEnum("serviceTier", o.ServiceTier, serviceTiers),
		validateEnum("sharedRequestType", o.SharedRequestType, sharedRequestTypes),
		validateEnum("requestType", o.RequestType, requestTypes),
	)
}

func (s SafetySetting) validate() error {
	if !slices.Contains(harmCategories, s.Category) {
		return fmt.Errorf("%w: safetySettings.category %q is not supported", model.ErrInvalidArgument, s.Category)
	}

	if s.Threshold == "" {
		return fmt.Errorf("%w: safetySettings.threshold is required", model.ErrInvalidArgument)
	}

	return validateEnum("safetySettings.threshold", s.Threshold, safetyThresholds)
}

func (c *ImageConfig) validate() error {
	if c == nil {
		return nil
	}

	errs := []error{
		validateEnum("imageConfig.aspectRatio", c.AspectRatio, aspectRatios),
		validateEnum("imageConfig.imageSize", c.ImageSize, imageSizes),
		validateEnum("imageConfig.personGeneration", c.PersonGeneration, personGenerations),
		validateEnum("imageConfig.prominentPeople", c.ProminentPeople, prominentPeopleSettings),
	}

	if c.ImageOutputOptions != nil {
		errs = append(errs, validateEnum("imageConfig.imageOutputOptions.mimeType", c.ImageOutputOptions.MimeType, imageOutputMimeTypes))
	}

	return errors.Join(errs...)
}

func (c *RetrievalConfig) validate() error {
	if c == nil || c.LatLng == nil {
		return nil
	}

	if c.LatLng.Latitude == nil || c.LatLng.Longitude == nil {
		return fmt.Errorf("%w: retrievalConfig.latLng requires latitude and longitude", model.ErrInvalidArgument)
	}

	return nil
}

// validateEnum accepts the empty value, which leaves the setting unset, and
// any of the allowed values.
func validateEnum[T ~string](field string, value T, allowed []T) error {
	if value == "" || slices.Contains(allowed, value) {
		return nil
	}

	return fmt.Errorf("%w: %s %q is not supported", model.ErrInvalidArgument, field, value)
}
