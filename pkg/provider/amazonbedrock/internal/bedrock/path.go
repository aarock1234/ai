package bedrock

import (
	"fmt"
	"strings"
)

// ModelPath returns the path of an action on a model, such as
// "/model/{modelId}/converse". The model ID is encoded like JavaScript's
// encodeURIComponent, so that the colons of versioned IDs are escaped.
func ModelPath(modelID string, action string) string {
	return "/model/" + encodeURIComponent(modelID) + "/" + action
}

func encodeURIComponent(value string) string {
	var encoded strings.Builder

	for index := range len(value) {
		character := value[index]

		if isUnescaped(character) {
			encoded.WriteByte(character)

			continue
		}

		fmt.Fprintf(&encoded, "%%%02X", character)
	}

	return encoded.String()
}

// isUnescaped reports the characters encodeURIComponent leaves unchanged.
func isUnescaped(character byte) bool {
	switch {
	case 'A' <= character && character <= 'Z':
		return true
	case 'a' <= character && character <= 'z':
		return true
	case '0' <= character && character <= '9':
		return true
	default:
		return strings.IndexByte("-_.!~*'()", character) >= 0
	}
}
