package azure

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

const (
	speechSynthesisPath = "/tts/cognitiveservices/v1"

	defaultVoice        = "en-US-Harper"
	defaultLocale       = "en-US"
	defaultOutputFormat = "audio-24khz-160kbitrate-mono-mp3"
)

// defaultVoices maps ISO 639-1 languages to their default voice, available
// on MAI-Voice-2, MAI-Voice-2.1 and their Flash variants.
var defaultVoices = map[string]string{
	"de": "de-DE-Mia",
	"en": defaultVoice,
	"es": "es-MX-Valeria",
	"fr": "fr-FR-Soleil",
	"hi": "hi-IN-Kavya",
	"hu": "hu-HU-Lilla",
	"it": "it-IT-Rosa",
	"ko": "ko-KR-Haena",
	"nl": "nl-NL-Fleur",
	"pt": "pt-BR-Luana",
	"ro": "ro-RO-Elena",
	"ru": "ru-RU-Masha",
	"th": "th-TH-Krit",
	"tr": "tr-TR-Elif",
	"zh": "zh-CN-Mei",
}

// shorthandOutputFormats maps output format shorthands to Azure Speech
// formats. Other X-Microsoft-OutputFormat values, such as
// "audio-48khz-192kbitrate-mono-mp3", are passed through.
var shorthandOutputFormats = map[string]string{
	"mp3":  defaultOutputFormat,
	"opus": "ogg-24khz-16bit-mono-opus",
	"pcm":  "raw-24khz-16bit-mono-pcm",
	"wav":  "riff-24khz-16bit-mono-pcm",
}

var (
	nativeOutputFormatPattern = regexp.MustCompile(`^(?:amr|audio|g722|ogg|raw|riff|webm)-[a-z0-9-]+$`)
	voiceLocalePattern        = regexp.MustCompile(`(?i)^([a-z]{2,3}-[a-z]{2,4})-`)
	voiceLanguagePattern      = regexp.MustCompile(`(?i)^([a-z]{2,3})-[a-z]{2,4}-`)
)

var xmlEscaper = strings.NewReplacer(
	"&", "&amp;",
	"<", "&lt;",
	">", "&gt;",
	`"`, "&quot;",
	"'", "&apos;",
)

// speechSynthesisModel converts text into speech with Azure Speech and SSML.
// The model is appended to the voice name, as in "en-US-Harper:MAI-Voice-2".
type speechSynthesisModel struct {
	modelID  string
	settings *settings
}

func (m *speechSynthesisModel) generate(ctx context.Context, options model.SpeechOptions, azureOptions SpeechOptions) (*model.SpeechResult, error) {
	timestamp := time.Now()

	var warnings []model.Warning

	if options.Instructions != "" {
		warnings = append(warnings, model.UnsupportedWarning("instructions", "Use providerOptions.azure.style to control speaking style."))
	}

	voice, languageWarning := resolveVoice(options.Voice, options.Language)
	if languageWarning != "" {
		warnings = append(warnings, model.UnsupportedWarning("language", languageWarning))
	}

	if azureOptions.StyleDegree != nil && azureOptions.Style == "" {
		warnings = append(warnings, model.UnsupportedWarning("providerOptions.azure.styleDegree", "styleDegree requires style."))
	}

	outputFormat, formatWarning := speechOutputFormat(options.OutputFormat)
	if formatWarning != nil {
		warnings = append(warnings, *formatWarning)
	}

	ssml := buildSSML(ssmlDocument{
		text:        options.Text,
		voiceName:   m.voiceName(voice),
		locale:      voiceLocale(voice),
		speed:       options.Speed,
		style:       azureOptions.Style,
		styleDegree: azureOptions.StyleDegree,
	})

	response, err := m.send(ctx, ssml, outputFormat, options)
	if err != nil {
		return nil, err
	}

	requestBody, err := json.Marshal(ssml)
	if err != nil {
		return nil, fmt.Errorf("encode ssml request body: %w", err)
	}

	return &model.SpeechResult{
		Audio:    response.Body,
		Warnings: warnings,
		Request: model.RequestInfo{
			Body: requestBody,
		},
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: timestamp,
				ModelID:   m.modelID,
			},
			Headers: response.Headers,
		},
	}, nil
}

// voiceName qualifies a voice with the model unless it already names one.
func (m *speechSynthesisModel) voiceName(voice string) string {
	if strings.Contains(voice, ":") {
		return voice
	}

	suffix, ok := maiVoiceModel(m.modelID)
	if !ok {
		suffix = m.modelID
	}

	return voice + ":" + suffix
}

// voiceLocale returns the locale prefix of a voice name, such as "en-US".
func voiceLocale(voice string) string {
	match := voiceLocalePattern.FindStringSubmatch(voice)
	if match == nil {
		return defaultLocale
	}

	return match[1]
}

// resolveVoice uses the language's default voice when no voice is set. An
// explicit voice wins, and its locale selects the language. The returned
// warning is empty when the language was honored.
func resolveVoice(voice string, language string) (string, string) {
	code := ""
	if language != "" {
		prefix, _, _ := strings.Cut(language, "-")
		code = strings.ToLower(prefix)
	}

	if voice == "" {
		return defaultLanguageVoice(code, language)
	}

	match := voiceLanguagePattern.FindStringSubmatch(voice)
	if code == "" || code == "auto" || match == nil || strings.ToLower(match[1]) == code {
		return voice, ""
	}

	return voice, fmt.Sprintf("The voice %s selects the language. Language %q was ignored.", voice, language)
}

// defaultLanguageVoice returns the default voice of a language code.
func defaultLanguageVoice(code string, language string) (string, string) {
	switch code {
	case "":
		return defaultVoice, ""
	case "auto":
		return defaultVoice, "Automatic language detection is not supported. " + defaultVoice + " was used."
	}

	if voice, ok := defaultVoices[code]; ok {
		return voice, ""
	}

	return defaultVoice, fmt.Sprintf("No default MAI voice for language %q. %s was used.", language, defaultVoice)
}

// speechOutputFormat maps an output format to an Azure Speech format,
// falling back to MP3 for unknown formats.
func speechOutputFormat(requested string) (string, *model.Warning) {
	if requested == "" {
		return defaultOutputFormat, nil
	}

	format := strings.ToLower(requested)
	if shorthand, ok := shorthandOutputFormats[format]; ok {
		return shorthand, nil
	}

	if nativeOutputFormatPattern.MatchString(format) {
		return format, nil
	}

	warning := model.UnsupportedWarning("outputFormat", "Unsupported output format: "+requested+". Using mp3 instead.")

	return defaultOutputFormat, &warning
}

// ssmlDocument holds the content of one SSML request.
type ssmlDocument struct {
	text        string
	voiceName   string
	locale      string
	speed       *float64
	style       string
	styleDegree *float64
}

// buildSSML renders the SSML document. Speed becomes a prosody rate and
// style an mstts:express-as element.
func buildSSML(document ssmlDocument) string {
	content := xmlEscaper.Replace(document.text)

	if document.speed != nil {
		content = fmt.Sprintf(`<prosody rate="%s">%s</prosody>`, formatNumber(*document.speed), content)
	}

	if document.style != "" {
		degree := ""
		if document.styleDegree != nil {
			degree = fmt.Sprintf(` styledegree="%s"`, formatNumber(*document.styleDegree))
		}

		content = fmt.Sprintf(`<mstts:express-as style="%s"%s>%s</mstts:express-as>`, xmlEscaper.Replace(document.style), degree, content)
	}

	return `<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" ` +
		`xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="` + xmlEscaper.Replace(document.locale) + `">` +
		`<voice name="` + xmlEscaper.Replace(document.voiceName) + `">` + content + `</voice></speak>`
}

func formatNumber(value float64) string {
	return strconv.FormatFloat(value, 'f', -1, 64)
}
