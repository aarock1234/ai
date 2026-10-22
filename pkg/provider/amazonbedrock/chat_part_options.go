package amazonbedrock

import (
	"fmt"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// CacheTTL is the lifetime of a prompt cache checkpoint.
type CacheTTL string

// Cache lifetimes.
const (
	// CacheTTL5Minutes is the default lifetime, supported by all models.
	CacheTTL5Minutes CacheTTL = "5m"

	// CacheTTL1Hour is supported by newer Claude models.
	CacheTTL1Hour CacheTTL = "1h"
)

// cachePointDefault is the only cache point type.
const cachePointDefault = "default"

// CachePoint is a prompt caching checkpoint. Set it as the cachePoint member
// of the "amazonBedrock" provider options of a system message, a message or
// a part to cache the prompt up to that point.
type CachePoint struct {
	// Type must be "default".
	Type string `json:"type"`

	// TTL is the cache lifetime. Empty uses CacheTTL5Minutes.
	TTL CacheTTL `json:"ttl,omitempty"`
}

// GuardContentQualifier marks how a guardrail evaluates guarded text.
type GuardContentQualifier string

// Guard content qualifiers.
const (
	GuardContentGroundingSource GuardContentQualifier = "grounding_source"
	GuardContentQuery           GuardContentQualifier = "query"
	GuardContentGuardContent    GuardContentQualifier = "guard_content"
)

// TextPartOptions are the "amazonBedrock" provider options of a user text
// part.
type TextPartOptions struct {
	// GuardContent sends the text as guardrail content.
	GuardContent *bool `json:"guardContent,omitempty"`

	GuardContentQualifiers []GuardContentQualifier `json:"guardContentQualifiers,omitempty"`
}

// ImagePartOptions are the "amazonBedrock" provider options of a user image
// part.
type ImagePartOptions struct {
	// GuardContent sends the image as guardrail content.
	GuardContent *bool `json:"guardContent,omitempty"`
}

// FilePartOptions are the "amazonBedrock" provider options of a document
// part.
type FilePartOptions struct {
	// Citations enables citations for the document.
	Citations *DocumentCitations `json:"citations,omitempty"`
}

// DocumentCitations configures the citations of a document.
type DocumentCitations struct {
	Enabled bool `json:"enabled"`
}

// cachePointOptions holds the cache point member of part options.
type cachePointOptions struct {
	CachePoint *CachePoint `json:"cachePoint"`
}

// cachePointOf returns the cache point of a message or part, read from
// "amazonBedrock" and then from "bedrock".
func cachePointOf(options model.ProviderOptions) (*CachePoint, error) {
	for _, namespace := range []string{Namespace, LegacyNamespace} {
		value, _, err := model.ProviderValue[cachePointOptions](options, namespace)
		if err != nil {
			return nil, err
		}

		if value.CachePoint == nil {
			continue
		}

		if err := value.CachePoint.validate(); err != nil {
			return nil, err
		}

		return value.CachePoint, nil
	}

	return nil, nil
}

func (c CachePoint) validate() error {
	if c.Type != cachePointDefault {
		return fmt.Errorf("%w: amazon bedrock cache point type %q", model.ErrInvalidArgument, c.Type)
	}

	if c.TTL == "" {
		return nil
	}

	return checkEnum("cachePoint.ttl", c.TTL, []CacheTTL{
		CacheTTL5Minutes,
		CacheTTL1Hour,
	})
}

func readTextPartOptions(options model.ProviderOptions) (TextPartOptions, error) {
	value, _, err := namespacedValue[TextPartOptions](options)
	if err != nil {
		return TextPartOptions{}, err
	}

	qualifiers := []GuardContentQualifier{
		GuardContentGroundingSource,
		GuardContentQuery,
		GuardContentGuardContent,
	}

	for _, qualifier := range value.GuardContentQualifiers {
		if !slices.Contains(qualifiers, qualifier) {
			return TextPartOptions{}, fmt.Errorf("%w: amazon bedrock guard content qualifier %q", model.ErrInvalidArgument, qualifier)
		}
	}

	return value, nil
}

func readImagePartOptions(options model.ProviderOptions) (ImagePartOptions, error) {
	value, _, err := namespacedValue[ImagePartOptions](options)

	return value, err
}

// citationsEnabled reports whether a document part enables citations.
func citationsEnabled(options model.ProviderOptions) (bool, error) {
	value, _, err := namespacedValue[FilePartOptions](options)
	if err != nil {
		return false, err
	}

	return value.Citations != nil && value.Citations.Enabled, nil
}
