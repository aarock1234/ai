package codemode

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/evanw/esbuild/pkg/api"
)

const (
	programName = "code-mode.js"

	// The program runs as the body of an async function, so that top-level
	// await and return work. The prefix shares the first line with the
	// program, so that reported line numbers match the program's.
	programPrefix = "(async () => {"
	programSuffix = "\n})()"
)

// syntaxMessagePattern matches syntax error messages that already say what
// they are; upstream prefixes the others with "Syntax error: ".
var syntaxMessagePattern = regexp.MustCompile(`(?i)syntax|unexpected|expression expected`)

// stripTypes wraps the program in its async function and strips its
// TypeScript syntax, as upstream does before running a program. The result
// carries an inline source map, so that the runtime reports locations in
// the program as written.
func stripTypes(js string) (string, error) {
	result := api.Transform(programPrefix+js+programSuffix, api.TransformOptions{
		Loader:         api.LoaderTS,
		Sourcefile:     programName,
		Sourcemap:      api.SourceMapInline,
		SourcesContent: api.SourcesContentExclude,
		Charset:        api.CharsetUTF8,
		LegalComments:  api.LegalCommentsNone,
		LogLevel:       api.LogLevelSilent,
	})

	if len(result.Errors) > 0 {
		return "", transformFailure(result.Errors[0])
	}

	return string(result.Code), nil
}

// transformFailure reports a program that does not parse as a SyntaxError
// whose message ends with its location in the program.
func transformFailure(message api.Message) *ScriptError {
	text := message.Text
	if !syntaxMessagePattern.MatchString(text) {
		text = "Syntax error: " + text
	}

	if message.Location != nil {
		text += " at " + programLocation(*message.Location)
	}

	return &ScriptError{
		Name:    "SyntaxError",
		Message: text,
	}
}

// programLocation converts a location in the wrapped program to a 1-based
// location in the program as written.
func programLocation(location api.Location) string {
	column := location.Column + 1

	if location.Line == 1 {
		column = max(1, column-len(programPrefix))
	}

	return fmt.Sprintf("%s:%d:%d", programName, location.Line, column)
}

// syntaxFailure reports a stripped program that the runtime does not
// compile. The message keeps the position, such as
// "Unexpected token } at code-mode.js:1:24".
func syntaxFailure(err error) error {
	text := strings.TrimPrefix(err.Error(), "SyntaxError: ")
	if !syntaxMessagePattern.MatchString(text) {
		text = "Syntax error: " + text
	}

	return &ScriptError{
		Name:    "SyntaxError",
		Message: text,
	}
}
