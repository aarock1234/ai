package anthropic

import (
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// maxCacheBreakpoints is the number of cache breakpoints Anthropic accepts
// per request.
const maxCacheBreakpoints = 4

// cacheControlCarrier reads a cache breakpoint from provider options, which
// may spell the key cacheControl or cache_control.
type cacheControlCarrier struct {
	CacheControl      *CacheControl `json:"cacheControl"`
	SnakeCacheControl *CacheControl `json:"cache_control"`
}

func (c cacheControlCarrier) value() *CacheControl {
	if c.CacheControl != nil {
		return c.CacheControl
	}

	return c.SnakeCacheControl
}

// cacheLocation describes where a cache breakpoint was requested.
type cacheLocation struct {
	description string
	canCache    bool
}

// cacheControlValidator resolves cache breakpoints across one request,
// enforcing the breakpoint limit. It records the first decode failure, which
// the request preparation reports after conversion.
type cacheControlValidator struct {
	namespace   string
	breakpoints int
	warnings    *callWarnings
	err         error
}

func newCacheControlValidator(namespace string, warnings *callWarnings) *cacheControlValidator {
	return &cacheControlValidator{
		namespace: namespace,
		warnings:  warnings,
	}
}

// cacheControl returns the breakpoint configured in options, preferring the
// provider's own namespace over "anthropic", or nil when none applies.
func (v *cacheControlValidator) cacheControl(options model.ProviderOptions, location cacheLocation) *CacheControl {
	cacheControl := v.lookup(options)
	if cacheControl == nil {
		return nil
	}

	if !location.canCache {
		v.warnings.unsupported(
			"cache_control on non-cacheable context",
			fmt.Sprintf("cache_control cannot be set on %s. It will be ignored.", location.description),
		)

		return nil
	}

	v.breakpoints++
	if v.breakpoints > maxCacheBreakpoints {
		v.warnings.unsupported(
			"cacheControl breakpoint limit",
			fmt.Sprintf(
				"Maximum %d cache breakpoints exceeded (found %d). This breakpoint will be ignored.",
				maxCacheBreakpoints,
				v.breakpoints,
			),
		)

		return nil
	}

	return cacheControl
}

func (v *cacheControlValidator) lookup(options model.ProviderOptions) *CacheControl {
	namespaces := []string{
		v.namespace,
		Namespace,
	}

	for _, namespace := range namespaces {
		carrier, found, err := model.ProviderValue[cacheControlCarrier](options, namespace)
		if err != nil {
			v.fail(err)

			return nil
		}

		if found && carrier.value() != nil {
			return carrier.value()
		}
	}

	return nil
}

func (v *cacheControlValidator) fail(err error) {
	if v.err == nil {
		v.err = err
	}
}

// Locations where cache breakpoints may be requested.
var (
	cacheSystemMessage = cacheLocation{
		description: "system message",
		canCache:    true,
	}
	cacheUserMessagePart = cacheLocation{
		description: "user message part",
		canCache:    true,
	}
	cacheUserMessage = cacheLocation{
		description: "user message",
		canCache:    true,
	}
	cacheToolResultPart = cacheLocation{
		description: "tool result part",
		canCache:    true,
	}
	cacheToolResultOutput = cacheLocation{
		description: "tool result output",
		canCache:    true,
	}
	cacheToolResultMessage = cacheLocation{
		description: "tool result message",
		canCache:    true,
	}
	cacheAssistantMessagePart = cacheLocation{
		description: "assistant message part",
		canCache:    true,
	}
	cacheAssistantMessage = cacheLocation{
		description: "assistant message",
		canCache:    true,
	}
	cacheThinkingBlock = cacheLocation{
		description: "thinking block",
		canCache:    false,
	}
	cacheRedactedThinkingBlock = cacheLocation{
		description: "redacted thinking block",
		canCache:    false,
	}
	cacheToolDefinition = cacheLocation{
		description: "tool definition",
		canCache:    true,
	}
)
