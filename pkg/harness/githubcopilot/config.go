package githubcopilot

import (
	"bytes"
	"encoding/json"
	"os"
)

// copilotConfig is the part of Copilot CLI's config.json that records its
// logins. Copilot has written both snake_case and camelCase keys. Each
// field stays raw, so a malformed field does not hide the others.
type copilotConfig struct {
	LastLoggedInUserSnake json.RawMessage `json:"last_logged_in_user"`
	LastLoggedInUser      json.RawMessage `json:"lastLoggedInUser"`
	LoggedInUsersSnake    json.RawMessage `json:"logged_in_users"`
	LoggedInUsers         json.RawMessage `json:"loggedInUsers"`

	// CopilotTokens are the plaintext tokens keyed by account, the fallback
	// without a credential store.
	CopilotTokens json.RawMessage `json:"copilotTokens"`
}

// copilotAccount is a Copilot CLI login: the origin of its GitHub host and
// its user login.
type copilotAccount struct {
	host  string
	login string
}

// key returns the account's key in the credential store and in the
// plaintext tokens.
func (a copilotAccount) key() string {
	return a.host + ":" + a.login
}

// readConfig reads Copilot CLI's JSON-with-comments configuration. A
// missing or malformed file yields the zero configuration.
func readConfig(configPath string) copilotConfig {
	content, err := os.ReadFile(configPath)
	if err != nil {
		return copilotConfig{}
	}

	var config copilotConfig
	if err := json.Unmarshal(stripJSONC(content), &config); err != nil {
		return copilotConfig{}
	}

	return config
}

// account returns the last logged-in account, falling back to the first
// valid logged-in account.
func (c copilotConfig) account() (copilotAccount, bool) {
	for _, value := range []json.RawMessage{
		c.LastLoggedInUserSnake,
		c.LastLoggedInUser,
	} {
		if account, ok := decodeAccount(value); ok {
			return account, true
		}
	}

	for _, list := range []json.RawMessage{
		c.LoggedInUsersSnake,
		c.LoggedInUsers,
	} {
		var values []json.RawMessage
		if err := json.Unmarshal(list, &values); err != nil {
			continue
		}

		for _, value := range values {
			if account, ok := decodeAccount(value); ok {
				return account, true
			}
		}
	}

	return copilotAccount{}, false
}

// decodeAccount decodes an account with a non-blank host and login whose
// host normalizes to an origin.
func decodeAccount(value json.RawMessage) (copilotAccount, bool) {
	var account struct {
		Host  string `json:"host"`
		Login string `json:"login"`
	}

	if err := json.Unmarshal(value, &account); err != nil {
		return copilotAccount{}, false
	}

	if !isNonBlank(account.Host) || !isNonBlank(account.Login) {
		return copilotAccount{}, false
	}

	origin, ok := normalizeGitHubOrigin(account.Host)
	if !ok {
		return copilotAccount{}, false
	}

	return copilotAccount{
		host:  origin,
		login: account.Login,
	}, true
}

// plaintextToken returns the non-blank plaintext token of an account.
func (c copilotConfig) plaintextToken(accountKey string) (string, bool) {
	var tokens map[string]json.RawMessage
	if err := json.Unmarshal(c.CopilotTokens, &tokens); err != nil {
		return "", false
	}

	var token string
	if err := json.Unmarshal(tokens[accountKey], &token); err != nil {
		return "", false
	}

	return token, isNonBlank(token)
}

// stripJSONC converts JSON with comments and trailing commas to JSON:
// comments become whitespace and trailing commas are removed. String
// contents are preserved.
func stripJSONC(content []byte) []byte {
	return removeTrailingCommas(removeComments(content))
}

// removeComments replaces line and block comments outside strings with
// spaces, keeping line breaks.
func removeComments(content []byte) []byte {
	result := bytes.Clone(content)
	inString := false

	for index := 0; index < len(result); index++ {
		character := result[index]

		if inString {
			switch character {
			case '\\':
				index++
			case '"':
				inString = false
			}

			continue
		}

		switch {
		case character == '"':
			inString = true
		case bytes.HasPrefix(result[index:], []byte("//")):
			end := bytes.IndexByte(result[index:], '\n')
			if end < 0 {
				end = len(result) - index
			}

			blank(result[index : index+end])
			index += end - 1
		case bytes.HasPrefix(result[index:], []byte("/*")):
			end := bytes.Index(result[index+2:], []byte("*/"))
			if end < 0 {
				blank(result[index:])

				return result
			}

			blank(result[index : index+2+end+2])
			index += 2 + end + 1
		}
	}

	return result
}

// blank replaces every character except line breaks with a space.
func blank(region []byte) {
	for index, character := range region {
		if character != '\n' && character != '\r' {
			region[index] = ' '
		}
	}
}

// removeTrailingCommas removes commas outside strings that only whitespace
// separates from a closing bracket or brace.
func removeTrailingCommas(content []byte) []byte {
	result := make([]byte, 0, len(content))
	inString := false

	for index := 0; index < len(content); index++ {
		character := content[index]

		if inString {
			result = append(result, character)

			switch character {
			case '\\':
				if index+1 < len(content) {
					index++
					result = append(result, content[index])
				}
			case '"':
				inString = false
			}

			continue
		}

		if character == '"' {
			inString = true
		}

		if character == ',' && closesAfterWhitespace(content[index+1:]) {
			continue
		}

		result = append(result, character)
	}

	return result
}

// closesAfterWhitespace reports whether the next non-whitespace character
// closes an array or object.
func closesAfterWhitespace(rest []byte) bool {
	trimmed := bytes.TrimLeft(rest, " \t\r\n")

	return len(trimmed) > 0 && (trimmed[0] == ']' || trimmed[0] == '}')
}
