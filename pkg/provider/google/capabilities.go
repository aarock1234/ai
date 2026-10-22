package google

import (
	"regexp"
	"strings"
)

var (
	gemini1ModelPattern        = regexp.MustCompile(`(?i)(^|[/.])gemini-1(?:[.-]|$)`)
	gemini2ModelPattern        = regexp.MustCompile(`(?i)(^|[/.])gemini-2(?:[.-]|$)`)
	gemini20ModelPattern       = regexp.MustCompile(`(^|/)gemini-2\.0`)
	gemini25ModelPattern       = regexp.MustCompile(`(?i)(^|[/.])gemini-2\.5(?:[.-]|$)`)
	geminiModelPattern         = regexp.MustCompile(`(?i)(^|[/.])gemini-`)
	geminiPrefixPattern        = regexp.MustCompile(`(?i)(^|/)gemini-`)
	externalURLModelPattern    = regexp.MustCompile(`(^|/)gemini-`)
	legacyGeminiProPattern     = regexp.MustCompile(`(?i)(^|[/.])gemini-pro(?:-vision)?$`)
	legacyRoboticsModelPattern = regexp.MustCompile(`(?i)(^|[/.])gemini-robotics-er-1\.5(?:[.-]|$)`)
)

// modelCapabilities classifies Gemini model IDs by excluding known older
// generations. Google model IDs are open-ended, so unrecognized Gemini IDs
// and aliases inherit the newest supported behavior.
type modelCapabilities struct {
	supportsGemini2Tools bool
	supportsFileSearch   bool
	usesGemini3Features  bool
}

func capabilitiesOf(modelID string) modelCapabilities {
	isGemini := geminiModelPattern.MatchString(modelID)
	isPreGemini2 := isKnownPreGemini2Model(modelID)
	isOlder := isPreGemini2 || gemini2ModelPattern.MatchString(modelID)
	usesGemini3Features := isGemini && !isOlder

	supportsGemini2Tools := isGemini && !isPreGemini2
	if strings.Contains(strings.ToLower(modelID), "nano-banana") {
		supportsGemini2Tools = true
	}

	return modelCapabilities{
		supportsGemini2Tools: supportsGemini2Tools,
		supportsFileSearch:   gemini25ModelPattern.MatchString(modelID) || usesGemini3Features,
		usesGemini3Features:  usesGemini3Features,
	}
}

func isKnownPreGemini2Model(modelID string) bool {
	if gemini1ModelPattern.MatchString(modelID) {
		return true
	}

	if legacyGeminiProPattern.MatchString(modelID) {
		return true
	}

	return legacyRoboticsModelPattern.MatchString(modelID)
}

func isGemmaModel(modelID string) bool {
	return strings.HasPrefix(strings.ToLower(modelID), "gemma-")
}

func isGeminiModel(modelID string) bool {
	return geminiPrefixPattern.MatchString(modelID)
}

// isGemini25Model reports Gemini 2.5 models, which reject frequency and
// presence penalties on the Gemini API.
func isGemini25Model(modelID string) bool {
	return gemini25ModelPattern.MatchString(modelID)
}

// supportsExternalFileURLs reports models that download arbitrary HTTPS file
// URLs themselves.
func supportsExternalFileURLs(modelID string) bool {
	return externalURLModelPattern.MatchString(modelID) && !gemini20ModelPattern.MatchString(modelID)
}

// modelPath returns the resource path of a model ID, which may already be a
// full resource name such as "tunedModels/my-model".
func modelPath(modelID string) string {
	if strings.Contains(modelID, "/") {
		return modelID
	}

	return "models/" + modelID
}
