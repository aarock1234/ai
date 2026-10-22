package middleware

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

const defaultInputExamplesPrefix = "Input Examples:"

// AddToolInputExamplesOptions configures AddToolInputExamples.
type AddToolInputExamplesOptions struct {
	// Prefix introduces the examples. The default is "Input Examples:".
	Prefix string

	// Format renders one example. The default renders compact JSON.
	Format func(example json.RawMessage, index int) string

	// KeepExamples keeps the InputExamples field after copying the examples
	// into the description. By default it is removed.
	KeepExamples bool
}

// AddToolInputExamples returns middleware that appends each function tool's
// input examples to its description, for providers that do not accept input
// examples natively.
func AddToolInputExamples(options AddToolInputExamplesOptions) Language {
	prefix := options.Prefix
	if prefix == "" {
		prefix = defaultInputExamplesPrefix
	}

	format := options.Format
	if format == nil {
		format = formatInputExample
	}

	return Language{
		TransformParams: func(_ context.Context, transform LanguageTransform) (model.CallOptions, error) {
			params := transform.Params
			if len(params.Tools) == 0 {
				return params, nil
			}

			tools := make([]model.Tool, len(params.Tools))
			for index, tool := range params.Tools {
				tools[index] = describeInputExamples(describeExamplesRequest{
					tool:         tool,
					prefix:       prefix,
					format:       format,
					keepExamples: options.KeepExamples,
				})
			}

			params.Tools = tools

			return params, nil
		},
	}
}

type describeExamplesRequest struct {
	tool         model.Tool
	prefix       string
	format       func(example json.RawMessage, index int) string
	keepExamples bool
}

func describeInputExamples(request describeExamplesRequest) model.Tool {
	function, ok := request.tool.(model.FunctionTool)
	if !ok || len(function.InputExamples) == 0 {
		return request.tool
	}

	formatted := make([]string, len(function.InputExamples))
	for index, example := range function.InputExamples {
		formatted[index] = request.format(example, index)
	}

	section := request.prefix + "\n" + strings.Join(formatted, "\n")

	if function.Description == "" {
		function.Description = section
	} else {
		function.Description = function.Description + "\n\n" + section
	}

	if !request.keepExamples {
		function.InputExamples = nil
	}

	return function
}

func formatInputExample(example json.RawMessage, _ int) string {
	var compacted bytes.Buffer
	if err := json.Compact(&compacted, example); err != nil {
		return string(example)
	}

	return compacted.String()
}
