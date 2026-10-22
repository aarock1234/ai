package openai

import (
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

var (
	oSeriesPattern = regexp.MustCompile(`^o(\d+)(?:-|$)`)
	gptPattern     = regexp.MustCompile(`^gpt-(\d+)(?:\.(\d+))?(?:-(.+))?$`)
)

// modelCapabilities describes how an OpenAI language model treats request
// parameters. Unrecognized model IDs, such as fine-tuned or third-party
// models, keep conservative defaults.
type modelCapabilities struct {
	isReasoningModel            bool
	systemMessageMode           ChatSystemMessageMode
	supportsFlexProcessing      bool
	supportsPriorityProcessing  bool
	supportsConfigurationUpdate bool
	supportsAsyncToolCalling    bool

	// supportedReasoningEfforts is nil when the model accepts every effort.
	supportedReasoningEfforts []model.ReasoningEffort

	// supportsNonReasoningParameters allows temperature, topP and logprobs
	// when the reasoning effort is "none".
	supportsNonReasoningParameters bool
}

// gptVersion is the parsed form of a "gpt-<major>[.<minor>][-<variant>]" ID.
type gptVersion struct {
	major    int
	minor    int
	hasMinor bool
	variant  string
}

// capabilitiesFor derives the capabilities of modelID from its name.
func capabilitiesFor(modelID string) modelCapabilities {
	oSeries, isOSeries := oSeriesVersion(modelID)
	gpt, isGPT := parseGPTVersion(modelID)

	isChatVariant := isGPT && !gpt.hasMinor && strings.HasPrefix(gpt.variant, "chat")
	isNanoVariant := isGPT && strings.HasPrefix(gpt.variant, "nano")
	isGPT5OrLater := isGPT && gpt.major >= 5 && !isChatVariant
	isGPT6OrLater := isGPT && gpt.major >= 6
	isGPT6SolOrLuna := modelID == "gpt-6-sol" || modelID == "gpt-6-luna"
	isModernOSeries := isOSeries && oSeries >= 3

	isReasoningModel := isOSeries || isGPT5OrLater

	systemMessageMode := ChatSystemMessageSystem
	if isReasoningModel {
		systemMessageMode = ChatSystemMessageDeveloper
	}

	isGPT4 := strings.HasPrefix(modelID, "gpt-4")
	supportsPriorityProcessing := isGPT4 || isModernOSeries || (isGPT5OrLater && !isNanoVariant)

	isGPT5Point1Family := isGPT && !isGPT6OrLater && supportsNoneEffortParameters(gpt)
	supportsNonReasoningParameters := isGPT6SolOrLuna || isGPT5Point1Family

	return modelCapabilities{
		isReasoningModel:               isReasoningModel,
		systemMessageMode:              systemMessageMode,
		supportsFlexProcessing:         isModernOSeries || isGPT5OrLater,
		supportsPriorityProcessing:     supportsPriorityProcessing,
		supportsConfigurationUpdate:    isGPT6OrLater,
		supportsAsyncToolCalling:       isGPT6OrLater,
		supportedReasoningEfforts:      supportedReasoningEfforts(isGPT6SolOrLuna, isGPT6OrLater),
		supportsNonReasoningParameters: supportsNonReasoningParameters,
	}
}

// supportsEffort reports whether the model accepts effort.
func (c modelCapabilities) supportsEffort(effort model.ReasoningEffort) bool {
	if c.supportedReasoningEfforts == nil {
		return true
	}

	return slices.Contains(c.supportedReasoningEfforts, effort)
}

func supportedReasoningEfforts(isGPT6SolOrLuna, isGPT6OrLater bool) []model.ReasoningEffort {
	switch {
	case isGPT6SolOrLuna:
		return []model.ReasoningEffort{
			model.ReasoningNone,
			model.ReasoningLow,
			model.ReasoningMedium,
			model.ReasoningHigh,
			model.ReasoningXHigh,
			model.ReasoningMax,
		}

	case isGPT6OrLater:
		return []model.ReasoningEffort{
			model.ReasoningLow,
			model.ReasoningMedium,
			model.ReasoningHigh,
			model.ReasoningXHigh,
			model.ReasoningMax,
		}

	default:
		return nil
	}
}

// supportsNoneEffortParameters reports whether a GPT model is 5.1 or later,
// which accepts sampling parameters when reasoning is disabled.
func supportsNoneEffortParameters(gpt gptVersion) bool {
	if gpt.major > 5 {
		return true
	}

	return gpt.major == 5 && gpt.minor >= 1
}

func oSeriesVersion(modelID string) (int, bool) {
	match := oSeriesPattern.FindStringSubmatch(modelID)
	if match == nil {
		return 0, false
	}

	version, err := strconv.Atoi(match[1])
	if err != nil {
		return 0, false
	}

	return version, true
}

func parseGPTVersion(modelID string) (gptVersion, bool) {
	match := gptPattern.FindStringSubmatch(modelID)
	if match == nil {
		return gptVersion{}, false
	}

	major, err := strconv.Atoi(match[1])
	if err != nil {
		return gptVersion{}, false
	}

	version := gptVersion{
		major:   major,
		variant: match[3],
	}

	if match[2] != "" {
		minor, err := strconv.Atoi(match[2])
		if err != nil {
			return gptVersion{}, false
		}

		version.minor = minor
		version.hasMinor = true
	}

	return version, true
}
