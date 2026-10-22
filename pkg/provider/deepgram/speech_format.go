package deepgram

import (
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

// audioSettings are the encoding, container, sample rate and bit rate query
// parameters of a speech request. Empty values are not sent.
type audioSettings struct {
	encoding   string
	container  string
	sampleRate string
	bitRate    string
}

// outputFormats map common output formats to valid Deepgram combinations.
var outputFormats = map[string]audioSettings{
	"mp3": {
		encoding: "mp3",
	},
	"wav": {
		encoding:  "linear16",
		container: "wav",
	},
	"linear16": {
		encoding:  "linear16",
		container: "wav",
	},
	"mulaw": {
		encoding:  "mulaw",
		container: "wav",
	},
	"alaw": {
		encoding:  "alaw",
		container: "wav",
	},
	"opus": {
		encoding:  "opus",
		container: "ogg",
	},
	"ogg": {
		encoding:  "opus",
		container: "ogg",
	},
	"flac": {
		encoding: "flac",
	},
	"aac": {
		encoding: "aac",
	},
	"pcm": {
		encoding:  "linear16",
		container: "none",
	},
}

var (
	// knownEncodings are the encodings recognized in "<encoding>_<rate>"
	// output formats.
	knownEncodings = []string{
		"linear16",
		"mulaw",
		"alaw",
		"mp3",
		"opus",
		"flac",
		"aac",
	}

	// containerEncodings support the wav and none containers.
	containerEncodings = []string{
		"linear16",
		"mulaw",
		"alaw",
	}

	// containerlessEncodings do not accept a container.
	containerlessEncodings = []string{
		"mp3",
		"flac",
		"aac",
	}

	// fixedRateEncodings do not accept a sample rate.
	fixedRateEncodings = []string{
		"mp3",
		"opus",
		"aac",
	}

	// losslessEncodings do not accept a bit rate.
	losslessEncodings = []string{
		"linear16",
		"mulaw",
		"alaw",
		"flac",
	}

	linear16SampleRates = []float64{
		8000,
		16000,
		24000,
		32000,
		48000,
	}

	telephonySampleRates = []float64{
		8000,
		16000,
	}

	flacSampleRates = []float64{
		8000,
		16000,
		22050,
		32000,
		48000,
	}

	mp3BitRates = []float64{
		32000,
		48000,
	}
)

// outputFormatSettings derives the audio settings of an output format, such
// as "wav", or "<encoding>_<rate>" and "<container>_<rate>" forms like
// "linear16_24000". Unknown formats leave every setting empty.
func outputFormatSettings(outputFormat string) audioSettings {
	format := strings.ToLower(outputFormat)

	if settings, ok := outputFormats[format]; ok {
		return settings
	}

	parts := strings.Split(format, "_")
	if len(parts) < 2 {
		return audioSettings{}
	}

	prefix := parts[0]
	sampleRate, hasRate := leadingInteger(parts[1])

	if slices.Contains(knownEncodings, prefix) {
		return encodingFormatSettings(prefix, sampleRate, hasRate)
	}

	var settings audioSettings

	switch prefix {
	case "wav":
		settings.container = "wav"
		settings.encoding = "linear16"
	case "ogg":
		settings.container = "ogg"
		settings.encoding = "opus"
	default:
		return audioSettings{}
	}

	if hasRate {
		settings.sampleRate = formatNumber(sampleRate)
	}

	return settings
}

// encodingFormatSettings derives the settings of an "<encoding>_<rate>"
// format. The rate is kept only when the encoding supports it.
func encodingFormatSettings(encoding string, sampleRate float64, hasRate bool) audioSettings {
	settings := audioSettings{
		encoding: encoding,
	}

	switch {
	case slices.Contains(containerEncodings, encoding):
		settings.container = "wav"
	case encoding == "opus":
		settings.container = "ogg"
	}

	if hasRate && supportedSampleRate(encoding, sampleRate) {
		settings.sampleRate = formatNumber(sampleRate)
	}

	return settings
}

func supportedSampleRate(encoding string, sampleRate float64) bool {
	switch encoding {
	case "linear16":
		return slices.Contains(linear16SampleRates, sampleRate)
	case "mulaw", "alaw":
		return slices.Contains(telephonySampleRates, sampleRate)
	case "flac":
		return slices.Contains(flacSampleRates, sampleRate)
	default:
		return false
	}
}

// applyOptions refines the settings with the provider options and returns
// warnings for the options that were ignored.
func (s *audioSettings) applyOptions(options SpeechOptions) []model.Warning {
	var warnings []model.Warning

	switch {
	case options.Encoding != nil && *options.Encoding != "":
		warnings = append(warnings, s.applyEncoding(*options.Encoding, options.Container)...)
	case options.Container != nil && *options.Container != "":
		s.applyContainer(*options.Container)
	}

	if options.SampleRate != nil {
		warnings = append(warnings, s.applySampleRate(*options.SampleRate)...)
	}

	if options.BitRate != nil {
		warnings = append(warnings, s.applyBitRate(*options.BitRate)...)
	}

	return warnings
}

// applyEncoding sets the encoding and keeps the container, sample rate and
// bit rate compatible with it.
func (s *audioSettings) applyEncoding(encoding string, container *string) []model.Warning {
	var warnings []model.Warning

	encoding = strings.ToLower(encoding)
	s.encoding = encoding

	if container != nil && *container != "" {
		warnings = append(warnings, s.applyEncodingContainer(encoding, *container)...)
	} else {
		switch {
		case slices.Contains(containerlessEncodings, encoding):
			s.container = ""
		case slices.Contains(containerEncodings, encoding):
			if s.container == "" {
				s.container = "wav"
			}
		case encoding == "opus":
			s.container = "ogg"
		}
	}

	s.removeIncompatibleRates(encoding)

	return warnings
}

// applyEncodingContainer applies an explicit container for the encoding.
func (s *audioSettings) applyEncodingContainer(encoding string, container string) []model.Warning {
	switch {
	case slices.Contains(containerEncodings, encoding):
		lower := strings.ToLower(container)
		if lower != "wav" && lower != "none" {
			return []model.Warning{
				optionsWarning(fmt.Sprintf(
					"Encoding %q only supports containers \"wav\" or \"none\". Container %q was ignored.",
					encoding,
					container,
				)),
			}
		}

		s.container = lower

	case encoding == "opus":
		s.container = "ogg"

	case slices.Contains(containerlessEncodings, encoding):
		s.container = ""

		return []model.Warning{
			optionsWarning(fmt.Sprintf(
				"Encoding %q does not support container parameter. Container %q was ignored.",
				encoding,
				container,
			)),
		}
	}

	return nil
}

// applyContainer sets a container without an explicit encoding, switching to
// the container's default encoding.
func (s *audioSettings) applyContainer(container string) {
	var encoding string

	switch strings.ToLower(container) {
	case "wav":
		s.container = "wav"
		encoding = "linear16"
	case "ogg":
		s.container = "ogg"
		encoding = "opus"
	case "none":
		s.container = "none"
		encoding = "linear16"
	default:
		return
	}

	if encoding == strings.ToLower(s.encoding) {
		return
	}

	s.encoding = encoding
	s.removeIncompatibleRates(encoding)
}

func (s *audioSettings) removeIncompatibleRates(encoding string) {
	if slices.Contains(fixedRateEncodings, encoding) {
		s.sampleRate = ""
	}

	if slices.Contains(losslessEncodings, encoding) {
		s.bitRate = ""
	}
}

func (s *audioSettings) applySampleRate(sampleRate float64) []model.Warning {
	encoding := strings.ToLower(s.encoding)
	text := formatNumber(sampleRate)

	switch {
	case encoding == "linear16" && !slices.Contains(linear16SampleRates, sampleRate):
		return []model.Warning{
			optionsWarning(fmt.Sprintf(
				"Encoding \"linear16\" only supports sample rates: 8000, 16000, 24000, 32000, 48000. Sample rate %s was ignored.",
				text,
			)),
		}

	case (encoding == "mulaw" || encoding == "alaw") && !slices.Contains(telephonySampleRates, sampleRate):
		return []model.Warning{
			optionsWarning(fmt.Sprintf(
				"Encoding %q only supports sample rates: 8000, 16000. Sample rate %s was ignored.",
				encoding,
				text,
			)),
		}

	case encoding == "flac" && !slices.Contains(flacSampleRates, sampleRate):
		return []model.Warning{
			optionsWarning(fmt.Sprintf(
				"Encoding \"flac\" only supports sample rates: 8000, 16000, 22050, 32000, 48000. Sample rate %s was ignored.",
				text,
			)),
		}

	case slices.Contains(fixedRateEncodings, encoding):
		return []model.Warning{
			optionsWarning(fmt.Sprintf(
				"Encoding %q has a fixed sample rate and does not support sample_rate parameter. Sample rate %s was ignored.",
				encoding,
				text,
			)),
		}
	}

	s.sampleRate = text

	return nil
}

func (s *audioSettings) applyBitRate(bitRate BitRate) []model.Warning {
	encoding := strings.ToLower(s.encoding)
	value := bitRateNumber(bitRate)

	switch {
	case encoding == "mp3" && !slices.Contains(mp3BitRates, value):
		return []model.Warning{
			optionsWarning(fmt.Sprintf(
				"Encoding \"mp3\" only supports bit rates: 32000, 48000. Bit rate %s was ignored.",
				bitRate,
			)),
		}

	case encoding == "opus" && (value < 4000 || value > 650000):
		return []model.Warning{
			optionsWarning(fmt.Sprintf(
				"Encoding \"opus\" supports bit rates between 4000 and 650000. Bit rate %s was ignored.",
				bitRate,
			)),
		}

	case encoding == "aac" && (value < 4000 || value > 192000):
		return []model.Warning{
			optionsWarning(fmt.Sprintf(
				"Encoding \"aac\" supports bit rates between 4000 and 192000. Bit rate %s was ignored.",
				bitRate,
			)),
		}

	case slices.Contains(losslessEncodings, encoding):
		return []model.Warning{
			optionsWarning(fmt.Sprintf(
				"Encoding %q does not support bit_rate parameter. Bit rate %s was ignored.",
				encoding,
				bitRate,
			)),
		}
	}

	s.bitRate = string(bitRate)

	return nil
}

func optionsWarning(details string) model.Warning {
	return model.UnsupportedWarning("providerOptions", details)
}

// bitRateNumber converts a bit rate as JavaScript's Number does: blank text
// is zero and malformed text is NaN, which fails every range check.
func bitRateNumber(bitRate BitRate) float64 {
	text := strings.TrimSpace(string(bitRate))
	if text == "" {
		return 0
	}

	value, err := strconv.ParseFloat(text, 64)
	if err != nil {
		return math.NaN()
	}

	return value
}

// leadingInteger parses the leading decimal digits of text, as JavaScript's
// parseInt does. It reports false when text does not start with a digit.
func leadingInteger(text string) (float64, bool) {
	end := strings.IndexFunc(text, func(r rune) bool {
		return r < '0' || r > '9'
	})
	if end < 0 {
		end = len(text)
	}

	value, err := strconv.ParseFloat(text[:end], 64)
	if err != nil {
		return 0, false
	}

	return value, true
}

func formatNumber(value float64) string {
	return strconv.FormatFloat(value, 'f', -1, 64)
}
