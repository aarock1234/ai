package model

import "encoding/json"

// Usage reports token consumption. Nil counts mean the provider did not
// report them.
type Usage struct {
	InputTokens  InputTokens
	OutputTokens OutputTokens

	// Raw is the provider's native usage object.
	Raw json.RawMessage
}

// InputTokens breaks down prompt tokens.
type InputTokens struct {
	Total *int

	// NoCache counts tokens that were neither read from nor written to a cache.
	NoCache *int

	CacheRead  *int
	CacheWrite *int
}

// OutputTokens breaks down completion tokens.
type OutputTokens struct {
	Total     *int
	Text      *int
	Reasoning *int
}
