package google

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// InteractionsOptions are the Google settings of an Interactions API call,
// read from ProviderOptions["google"].
type InteractionsOptions struct {
	// PreviousInteractionID chains the call to a stored interaction. Earlier
	// assistant turns of that interaction are then left out of the input.
	PreviousInteractionID *string `json:"previousInteractionId,omitempty"`

	// Store keeps the interaction on the server. False runs fully
	// stateless.
	Store *bool `json:"store,omitempty"`

	// AgentConfig configures an agent call.
	AgentConfig *InteractionsAgentConfig `json:"agentConfig,omitempty"`

	ThinkingLevel     InteractionsThinkingLevel     `json:"thinkingLevel,omitempty"`
	ThinkingSummaries InteractionsThinkingSummaries `json:"thinkingSummaries,omitempty"`

	// ResponseFormat entries are sent in order after the entry of a JSON
	// call-level response format. Use them to request image, audio, video
	// or non-JSON text output.
	ResponseFormat []InteractionsResponseFormat `json:"responseFormat,omitempty"`

	// ImageConfig requests image output.
	//
	// Deprecated: use a ResponseFormat entry of type "image".
	ImageConfig *InteractionsImageConfig `json:"imageConfig,omitempty"`

	MediaResolution    InteractionsMediaResolution    `json:"mediaResolution,omitempty"`
	ResponseModalities []InteractionsResponseModality `json:"responseModalities,omitempty"`
	ServiceTier        InteractionsServiceTier        `json:"serviceTier,omitempty"`

	// SystemInstruction is used when the prompt has no system message.
	SystemInstruction *string `json:"systemInstruction,omitempty"`

	// PollingTimeoutMs bounds the polling of a background interaction. The
	// default is 30 minutes.
	PollingTimeoutMs *int `json:"pollingTimeoutMs,omitempty"`

	// Background runs the interaction in the background, which some agents
	// require. Generate then polls until the interaction ends, and Stream
	// follows its event stream.
	Background *bool `json:"background,omitempty"`

	// Environment configures the sandbox of an agent call.
	Environment *InteractionsEnvironment `json:"environment,omitempty"`
}

// InteractionsThinkingLevel is the thinking depth of a model call.
type InteractionsThinkingLevel string

// Interactions thinking levels.
const (
	InteractionsThinkingMinimal InteractionsThinkingLevel = "minimal"
	InteractionsThinkingLow     InteractionsThinkingLevel = "low"
	InteractionsThinkingMedium  InteractionsThinkingLevel = "medium"
	InteractionsThinkingHigh    InteractionsThinkingLevel = "high"
)

// InteractionsThinkingSummaries selects whether thought summaries are
// returned.
type InteractionsThinkingSummaries string

// Interactions thinking summary modes.
const (
	InteractionsThinkingSummariesAuto InteractionsThinkingSummaries = "auto"
	InteractionsThinkingSummariesNone InteractionsThinkingSummaries = "none"
)

// InteractionsMediaResolution is the resolution of image and video input.
type InteractionsMediaResolution string

// Interactions media resolutions.
const (
	InteractionsMediaResolutionLow       InteractionsMediaResolution = "low"
	InteractionsMediaResolutionMedium    InteractionsMediaResolution = "medium"
	InteractionsMediaResolutionHigh      InteractionsMediaResolution = "high"
	InteractionsMediaResolutionUltraHigh InteractionsMediaResolution = "ultra_high"
)

// InteractionsResponseModality is an output modality.
type InteractionsResponseModality string

// Interactions response modalities.
const (
	InteractionsModalityText     InteractionsResponseModality = "text"
	InteractionsModalityImage    InteractionsResponseModality = "image"
	InteractionsModalityAudio    InteractionsResponseModality = "audio"
	InteractionsModalityVideo    InteractionsResponseModality = "video"
	InteractionsModalityDocument InteractionsResponseModality = "document"
)

// InteractionsServiceTier is the processing tier of a call.
type InteractionsServiceTier string

// Interactions service tiers.
const (
	InteractionsServiceTierFlex     InteractionsServiceTier = "flex"
	InteractionsServiceTierStandard InteractionsServiceTier = "standard"
	InteractionsServiceTierPriority InteractionsServiceTier = "priority"
)

// InteractionsAgentConfigType selects the agent configuration.
type InteractionsAgentConfigType string

// Interactions agent configuration types.
const (
	InteractionsAgentDynamic      InteractionsAgentConfigType = "dynamic"
	InteractionsAgentDeepResearch InteractionsAgentConfigType = "deep-research"
)

// InteractionsAgentConfig configures an agent call. ThinkingSummaries,
// Visualization and CollaborativePlanning apply to deep research.
type InteractionsAgentConfig struct {
	Type InteractionsAgentConfigType `json:"type"`

	ThinkingSummaries InteractionsThinkingSummaries `json:"thinkingSummaries,omitempty"`

	// Visualization is "off" or "auto".
	Visualization string `json:"visualization,omitempty"`

	CollaborativePlanning *bool `json:"collaborativePlanning,omitempty"`
}

// InteractionsResponseFormatType is the modality of a response format
// entry.
type InteractionsResponseFormatType string

// Interactions response format types.
const (
	InteractionsFormatText  InteractionsResponseFormatType = "text"
	InteractionsFormatImage InteractionsResponseFormatType = "image"
	InteractionsFormatAudio InteractionsResponseFormatType = "audio"
	InteractionsFormatVideo InteractionsResponseFormatType = "video"
)

// InteractionsResponseFormat is one requested output format. MimeType and
// Schema apply to text; MimeType, AspectRatio and ImageSize to images;
// MimeType to audio; and AspectRatio, Resolution, Duration, Delivery and
// GCSURI to video.
type InteractionsResponseFormat struct {
	Type InteractionsResponseFormatType `json:"type"`

	MimeType string          `json:"mimeType,omitempty"`
	Schema   json.RawMessage `json:"schema,omitempty"`

	// AspectRatio is an image ratio such as "16:9", or "16:9" or "9:16" for
	// video.
	AspectRatio string `json:"aspectRatio,omitempty"`

	// ImageSize is "1K", "2K", "4K" or "512".
	ImageSize string `json:"imageSize,omitempty"`

	// Resolution is "360p", "720p", "1080p" or "4k".
	Resolution string `json:"resolution,omitempty"`

	// Duration is a video length such as "8s".
	Duration string `json:"duration,omitempty"`

	// Delivery is "inline" or "uri".
	Delivery string `json:"delivery,omitempty"`

	// GCSURI is the Cloud Storage destination of a video.
	GCSURI string `json:"gcsUri,omitempty"`
}

// InteractionsImageConfig requests image output.
//
// Deprecated: use a ResponseFormat entry of type "image".
type InteractionsImageConfig struct {
	AspectRatio string `json:"aspectRatio,omitempty"`
	ImageSize   string `json:"imageSize,omitempty"`
}

// InteractionsEnvironment is the sandbox of an agent call: an ID, where
// "remote" provisions a fresh sandbox and any other ID reuses an existing
// one, or a remote sandbox preloaded with Sources and constrained by
// Network.
type InteractionsEnvironment struct {
	// ID is "remote" or the ID of an existing environment. It is exclusive
	// with Sources and Network.
	ID string

	Sources []InteractionsEnvironmentSource

	// Network is nil for the default network access.
	Network *InteractionsNetwork
}

// interactionsRemoteEnvironment is the object form of an environment.
type interactionsRemoteEnvironment struct {
	Type    string                          `json:"type"`
	Sources []InteractionsEnvironmentSource `json:"sources,omitempty"`
	Network *InteractionsNetwork            `json:"network,omitempty"`
}

// MarshalJSON encodes the ID as a string, and a sandbox as an object.
func (e InteractionsEnvironment) MarshalJSON() ([]byte, error) {
	if e.ID != "" {
		return json.Marshal(e.ID)
	}

	return json.Marshal(interactionsRemoteEnvironment{
		Type:    "remote",
		Sources: e.Sources,
		Network: e.Network,
	})
}

// UnmarshalJSON decodes a string ID or a remote sandbox object.
func (e *InteractionsEnvironment) UnmarshalJSON(data []byte) error {
	var id string
	if err := json.Unmarshal(data, &id); err == nil {
		*e = InteractionsEnvironment{
			ID: id,
		}

		return nil
	}

	var remote interactionsRemoteEnvironment
	if err := json.Unmarshal(data, &remote); err != nil {
		return err
	}

	if remote.Type != "remote" {
		return fmt.Errorf("environment type must be %q, got %q", "remote", remote.Type)
	}

	*e = InteractionsEnvironment{
		Sources: remote.Sources,
		Network: remote.Network,
	}

	return nil
}

// InteractionsEnvironmentSourceType is the kind of an environment source.
type InteractionsEnvironmentSourceType string

// Interactions environment source types.
const (
	InteractionsSourceGCS        InteractionsEnvironmentSourceType = "gcs"
	InteractionsSourceRepository InteractionsEnvironmentSourceType = "repository"
	InteractionsSourceInline     InteractionsEnvironmentSourceType = "inline"
)

// InteractionsEnvironmentSource preloads the sandbox: a Cloud Storage or
// repository Source copied to an optional Target, or inline Content written
// to Target.
type InteractionsEnvironmentSource struct {
	Type    InteractionsEnvironmentSourceType `json:"type"`
	Source  string                            `json:"source,omitempty"`
	Content string                            `json:"content,omitempty"`
	Target  string                            `json:"target,omitempty"`
}

// InteractionsNetwork constrains the outbound traffic of the sandbox:
// Disabled turns it off, and Allowlist otherwise lists the reachable
// domains.
type InteractionsNetwork struct {
	Disabled  bool
	Allowlist []InteractionsAllowlistEntry
}

type interactionsNetworkAllowlist struct {
	Allowlist []InteractionsAllowlistEntry `json:"allowlist"`
}

// MarshalJSON encodes "disabled" or the allowlist object.
func (n InteractionsNetwork) MarshalJSON() ([]byte, error) {
	if n.Disabled {
		return json.Marshal("disabled")
	}

	allowlist := n.Allowlist
	if allowlist == nil {
		allowlist = []InteractionsAllowlistEntry{}
	}

	return json.Marshal(interactionsNetworkAllowlist{
		Allowlist: allowlist,
	})
}

// UnmarshalJSON decodes "disabled" or the allowlist object.
func (n *InteractionsNetwork) UnmarshalJSON(data []byte) error {
	var mode string
	if err := json.Unmarshal(data, &mode); err == nil {
		if mode != "disabled" {
			return fmt.Errorf("network must be %q or an allowlist, got %q", "disabled", mode)
		}

		*n = InteractionsNetwork{
			Disabled: true,
		}

		return nil
	}

	var allowlist interactionsNetworkAllowlist
	if err := json.Unmarshal(data, &allowlist); err != nil {
		return err
	}

	*n = InteractionsNetwork{
		Allowlist: allowlist.Allowlist,
	}

	return nil
}

// InteractionsAllowlistEntry is one reachable domain, with optional request
// transforms.
type InteractionsAllowlistEntry struct {
	Domain    string              `json:"domain"`
	Transform []map[string]string `json:"transform,omitempty"`
}

// InteractionsVideoOptions are the Google settings of a video file part,
// read from the part's ProviderOptions["google"].
type InteractionsVideoOptions struct {
	Processing *InteractionsVideoProcessing `json:"processing,omitempty"`
}

// InteractionsVideoProcessingMode selects how Gemini processes a video.
type InteractionsVideoProcessingMode string

// Interactions video processing modes.
const (
	// InteractionsVideoAgentic explores the video timeline dynamically.
	InteractionsVideoAgentic InteractionsVideoProcessingMode = "agentic"

	// InteractionsVideoStatic samples frames at a fixed rate.
	InteractionsVideoStatic InteractionsVideoProcessingMode = "static"
)

// InteractionsVideoProcessing configures video processing. Static
// processing optionally clips the video, with offsets in seconds, and sets
// the frame rate.
type InteractionsVideoProcessing struct {
	Mode InteractionsVideoProcessingMode

	StartOffset *float64
	EndOffset   *float64
	FPS         *float64
}

// interactionsStaticProcessing is the object form of static processing.
type interactionsStaticProcessing struct {
	Type        string   `json:"type"`
	StartOffset *float64 `json:"startOffset,omitempty"`
	EndOffset   *float64 `json:"endOffset,omitempty"`
	FPS         *float64 `json:"fps,omitempty"`
}

// MarshalJSON encodes the mode as a string, or static processing with
// settings as an object.
func (p InteractionsVideoProcessing) MarshalJSON() ([]byte, error) {
	if p.StartOffset == nil && p.EndOffset == nil && p.FPS == nil {
		return json.Marshal(p.Mode)
	}

	return json.Marshal(interactionsStaticProcessing{
		Type:        string(InteractionsVideoStatic),
		StartOffset: p.StartOffset,
		EndOffset:   p.EndOffset,
		FPS:         p.FPS,
	})
}

// interactionsPartOptions are the Google settings and metadata read back
// from a prompt part.
type interactionsPartOptions struct {
	Signature        *string `json:"signature"`
	InteractionID    *string `json:"interactionId"`
	ProcessingID     *string `json:"processingId"`
	ProcessingCallID *string `json:"processingCallId"`

	// Processing is decoded leniently: an invalid value drops the option
	// with a warning.
	Processing json.RawMessage `json:"processing"`
}

var (
	interactionsThinkingLevels = []InteractionsThinkingLevel{
		InteractionsThinkingMinimal,
		InteractionsThinkingLow,
		InteractionsThinkingMedium,
		InteractionsThinkingHigh,
	}

	interactionsThinkingSummaryModes = []InteractionsThinkingSummaries{
		InteractionsThinkingSummariesAuto,
		InteractionsThinkingSummariesNone,
	}

	interactionsMediaResolutions = []InteractionsMediaResolution{
		InteractionsMediaResolutionLow,
		InteractionsMediaResolutionMedium,
		InteractionsMediaResolutionHigh,
		InteractionsMediaResolutionUltraHigh,
	}

	interactionsServiceTiers = []InteractionsServiceTier{
		InteractionsServiceTierFlex,
		InteractionsServiceTierStandard,
		InteractionsServiceTierPriority,
	}

	interactionsResponseModalities = []InteractionsResponseModality{
		InteractionsModalityText,
		InteractionsModalityImage,
		InteractionsModalityAudio,
		InteractionsModalityVideo,
		InteractionsModalityDocument,
	}

	interactionsAgentConfigTypes = []InteractionsAgentConfigType{
		InteractionsAgentDynamic,
		InteractionsAgentDeepResearch,
	}

	interactionsResponseFormatTypes = []InteractionsResponseFormatType{
		InteractionsFormatText,
		InteractionsFormatImage,
		InteractionsFormatAudio,
		InteractionsFormatVideo,
	}
)

// readInteractionsOptions reads and validates the call's Google options.
func readInteractionsOptions(providerOptions model.ProviderOptions) (InteractionsOptions, error) {
	options, _, err := model.ProviderValue[InteractionsOptions](providerOptions, Namespace)
	if err != nil {
		return InteractionsOptions{}, err
	}

	if err := options.validate(); err != nil {
		return InteractionsOptions{}, err
	}

	return options, nil
}

func (o InteractionsOptions) validate() error {
	checks := []error{
		validateEnum("thinkingLevel", o.ThinkingLevel, interactionsThinkingLevels),
		validateEnum("thinkingSummaries", o.ThinkingSummaries, interactionsThinkingSummaryModes),
		validateEnum("mediaResolution", o.MediaResolution, interactionsMediaResolutions),
		validateEnum("serviceTier", o.ServiceTier, interactionsServiceTiers),
	}

	for _, modality := range o.ResponseModalities {
		checks = append(checks, validateEnum("responseModalities", modality, interactionsResponseModalities))
	}

	for _, entry := range o.ResponseFormat {
		checks = append(checks, requireEnum("responseFormat.type", entry.Type, interactionsResponseFormatTypes))
	}

	if o.AgentConfig != nil {
		checks = append(checks, requireEnum("agentConfig.type", o.AgentConfig.Type, interactionsAgentConfigTypes))
	}

	if o.PollingTimeoutMs != nil && *o.PollingTimeoutMs <= 0 {
		checks = append(checks, fmt.Errorf("%w: google pollingTimeoutMs must be positive", model.ErrInvalidArgument))
	}

	for _, err := range checks {
		if err != nil {
			return err
		}
	}

	return nil
}

// requireEnum accepts only the allowed values.
func requireEnum[T ~string](field string, value T, allowed []T) error {
	if value == "" {
		return fmt.Errorf("%w: %s is required", model.ErrInvalidArgument, field)
	}

	return validateEnum(field, value, allowed)
}
