package gateway

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// Gateway tool IDs. The gateway executes these tools itself.
const (
	browserbaseFetchToolID  = "gateway.browserbase_fetch"
	browserbaseSearchToolID = "gateway.browserbase_search"
	exaSearchToolID         = "gateway.exa_search"
	parallelSearchToolID    = "gateway.parallel_search"
	perplexitySearchToolID  = "gateway.perplexity_search"
	takoSearchToolID        = "gateway.tako_search"
)

// BrowserbaseFetchFormat is the output format of fetched content.
type BrowserbaseFetchFormat string

// Browserbase fetch formats.
const (
	BrowserbaseFetchRaw      BrowserbaseFetchFormat = "raw"
	BrowserbaseFetchJSON     BrowserbaseFetchFormat = "json"
	BrowserbaseFetchMarkdown BrowserbaseFetchFormat = "markdown"
)

// BrowserbaseFetchArgs configure the Browserbase fetch tool.
type BrowserbaseFetchArgs struct {
	// AllowRedirects follows HTTP redirects. The default is false.
	AllowRedirects *bool `json:"allowRedirects,omitempty"`

	// AllowInsecureSSL skips TLS certificate verification. The default is
	// false.
	AllowInsecureSSL *bool `json:"allowInsecureSsl,omitempty"`

	// Proxies routes requests through Browserbase proxies. The default is
	// false.
	Proxies *bool `json:"proxies,omitempty"`

	// Format is the output format. The default is raw.
	Format BrowserbaseFetchFormat `json:"format,omitempty"`

	// Schema is the JSON Schema of the content, used with the JSON format.
	Schema json.RawMessage `json:"schema,omitempty"`
}

// BrowserbaseSearchArgs configure the Browserbase search tool.
type BrowserbaseSearchArgs struct {
	// NumResults is the default result count, from 1 to 25. The default is
	// 10.
	NumResults *int `json:"numResults,omitempty"`
}

// ExaSearchType is an Exa search method.
type ExaSearchType string

// Exa search methods.
const (
	ExaSearchAuto    ExaSearchType = "auto"
	ExaSearchFast    ExaSearchType = "fast"
	ExaSearchInstant ExaSearchType = "instant"
)

// ExaSearchArgs configure the Exa search tool.
type ExaSearchArgs struct {
	// Type is the default search method. Exa defaults to auto.
	Type ExaSearchType `json:"type,omitempty"`

	// NumResults is the default result count, from 1 to 100. The default is
	// 10.
	NumResults *int `json:"numResults,omitempty"`

	// Category filters results, such as "news" or "research paper".
	Category string `json:"category,omitempty"`

	// UserLocation is a two-letter ISO country code.
	UserLocation string `json:"userLocation,omitempty"`

	IncludeDomains []string `json:"includeDomains,omitempty"`
	ExcludeDomains []string `json:"excludeDomains,omitempty"`

	// StartPublishedDate and EndPublishedDate filter by ISO 8601 date.
	StartPublishedDate string `json:"startPublishedDate,omitempty"`
	EndPublishedDate   string `json:"endPublishedDate,omitempty"`

	// Contents controls content extraction.
	Contents *ExaSearchContents `json:"contents,omitempty"`
}

// ExaSearchContents controls content extraction.
type ExaSearchContents struct {
	Text       *ExaTextContents       `json:"text,omitempty"`
	Highlights *ExaHighlightsContents `json:"highlights,omitempty"`

	MaxAgeHours      *float64 `json:"maxAgeHours,omitempty"`
	LivecrawlTimeout *float64 `json:"livecrawlTimeout,omitempty"`
	Subpages         *int     `json:"subpages,omitempty"`
	SubpageTarget    []string `json:"subpageTarget,omitempty"`

	Extras *ExaExtras `json:"extras,omitempty"`
}

// ExaTextContents enables page text, optionally with settings. It encodes
// as true or false, or as the settings when Settings is set.
type ExaTextContents struct {
	Enabled  bool
	Settings *ExaTextSettings
}

// MarshalJSON writes the settings, or whether text is enabled.
func (c ExaTextContents) MarshalJSON() ([]byte, error) {
	if c.Settings != nil {
		return json.Marshal(*c.Settings)
	}

	return json.Marshal(c.Enabled)
}

// ExaTextSettings configure extracted page text.
type ExaTextSettings struct {
	MaxCharacters   *int   `json:"maxCharacters,omitempty"`
	IncludeHTMLTags *bool  `json:"includeHtmlTags,omitempty"`
	Verbosity       string `json:"verbosity,omitempty"`

	// IncludeSections and ExcludeSections are page sections such as
	// "body" or "navigation".
	IncludeSections []string `json:"includeSections,omitempty"`
	ExcludeSections []string `json:"excludeSections,omitempty"`
}

// ExaHighlightsContents enables highlights, optionally with settings. It
// encodes as true or false, or as the settings when Settings is set.
type ExaHighlightsContents struct {
	Enabled  bool
	Settings *ExaHighlightsSettings
}

// MarshalJSON writes the settings, or whether highlights are enabled.
func (c ExaHighlightsContents) MarshalJSON() ([]byte, error) {
	if c.Settings != nil {
		return json.Marshal(*c.Settings)
	}

	return json.Marshal(c.Enabled)
}

// ExaHighlightsSettings configure highlights.
type ExaHighlightsSettings struct {
	Query         string `json:"query,omitempty"`
	MaxCharacters *int   `json:"maxCharacters,omitempty"`
}

// ExaExtras requests links from each page.
type ExaExtras struct {
	Links      *int `json:"links,omitempty"`
	ImageLinks *int `json:"imageLinks,omitempty"`
}

// ParallelSearchMode is a Parallel search preset.
type ParallelSearchMode string

// Parallel search modes.
const (
	// ParallelOneShot returns comprehensive results with longer excerpts.
	// It is the default.
	ParallelOneShot ParallelSearchMode = "one-shot"

	// ParallelAgentic returns concise results for multi-step workflows.
	ParallelAgentic ParallelSearchMode = "agentic"
)

// ParallelSearchArgs configure the Parallel search tool.
type ParallelSearchArgs struct {
	Mode ParallelSearchMode `json:"mode,omitempty"`

	// MaxResults is the default result count, from 1 to 20. The default is
	// 10.
	MaxResults *int `json:"maxResults,omitempty"`

	SourcePolicy *ParallelSourcePolicy `json:"sourcePolicy,omitempty"`
	Excerpts     *ParallelExcerpts     `json:"excerpts,omitempty"`
	FetchPolicy  *ParallelFetchPolicy  `json:"fetchPolicy,omitempty"`
}

// ParallelSourcePolicy selects the domains and dates of results.
type ParallelSourcePolicy struct {
	IncludeDomains []string `json:"includeDomains,omitempty"`
	ExcludeDomains []string `json:"excludeDomains,omitempty"`

	// AfterDate keeps results published after an ISO 8601 date.
	AfterDate string `json:"afterDate,omitempty"`
}

// ParallelExcerpts limit excerpt length.
type ParallelExcerpts struct {
	MaxCharsPerResult *int `json:"maxCharsPerResult,omitempty"`
	MaxCharsTotal     *int `json:"maxCharsTotal,omitempty"`
}

// ParallelFetchPolicy controls content freshness.
type ParallelFetchPolicy struct {
	// MaxAgeSeconds is the maximum cache age; zero always fetches.
	MaxAgeSeconds *int `json:"maxAgeSeconds,omitempty"`
}

// PerplexitySearchArgs configure the Perplexity search tool.
type PerplexitySearchArgs struct {
	// MaxResults is the default result count, from 1 to 20. The default is
	// 10.
	MaxResults *int `json:"maxResults,omitempty"`

	// MaxTokensPerPage is the per-page token limit, from 256 to 2048.
	MaxTokensPerPage *int `json:"maxTokensPerPage,omitempty"`

	// MaxTokens is the total token limit, up to 1000000.
	MaxTokens *int `json:"maxTokens,omitempty"`

	// Country is a two-letter ISO 3166-1 country code.
	Country string `json:"country,omitempty"`

	// SearchDomainFilter includes domains, or excludes them with a "-"
	// prefix.
	SearchDomainFilter []string `json:"searchDomainFilter,omitempty"`

	// SearchLanguageFilter lists ISO 639-1 language codes.
	SearchLanguageFilter []string `json:"searchLanguageFilter,omitempty"`

	// SearchRecencyFilter is "day", "week", "month" or "year".
	SearchRecencyFilter string `json:"searchRecencyFilter,omitempty"`
}

// TakoSearchArgs configure the Tako search tool.
type TakoSearchArgs struct {
	// Effort is "deep", "fast" or "instant". The default is fast.
	Effort string `json:"effort,omitempty"`

	// Sources selects the sources to search. Nil searches curated data and
	// the web.
	Sources *TakoSources `json:"sources,omitempty"`

	// Location is the end user's coordinates.
	Location *TakoLocation `json:"location,omitempty"`

	CountryCode string `json:"countryCode,omitempty"`
	Locale      string `json:"locale,omitempty"`
	Timezone    string `json:"timezone,omitempty"`

	OutputSettings *TakoOutputSettings `json:"outputSettings,omitempty"`

	// IncludeRelated is the number of related search suggestions, from 1 to
	// 20.
	IncludeRelated *int `json:"includeRelated,omitempty"`
}

// TakoSources configure the data and web sources.
type TakoSources struct {
	Data *TakoDataSource `json:"data,omitempty"`
	Web  *TakoWebSource  `json:"web,omitempty"`
}

// TakoDataSource configures curated data results. Inlined contents add a
// data surcharge.
type TakoDataSource struct {
	Count           *int   `json:"count,omitempty"`
	IncludeContents *bool  `json:"includeContents,omitempty"`
	Mode            string `json:"mode,omitempty"`

	// ContentFormat is "card_json", "csv", "json_compact" or
	// "json_records".
	ContentFormat string `json:"contentFormat,omitempty"`

	MaxRows *int     `json:"maxRows,omitempty"`
	NodeIDs []string `json:"nodeIds,omitempty"`
	Strict  *bool    `json:"strict,omitempty"`
}

// TakoWebSource configures web results.
type TakoWebSource struct {
	Count                  *int     `json:"count,omitempty"`
	IncludeContents        *bool    `json:"includeContents,omitempty"`
	Category               string   `json:"category,omitempty"`
	IncludeDomains         []string `json:"includeDomains,omitempty"`
	ExcludeDomains         []string `json:"excludeDomains,omitempty"`
	SnippetMaxChars        *int     `json:"snippetMaxChars,omitempty"`
	Highlights             *bool    `json:"highlights,omitempty"`
	ArticleContentMaxChars *int     `json:"articleContentMaxChars,omitempty"`
	PublishedAfter         string   `json:"publishedAfter,omitempty"`
	PublishedBefore        string   `json:"publishedBefore,omitempty"`
}

// TakoLocation is a pair of coordinates.
type TakoLocation struct {
	Latitude  float64 `json:"latitude"`
	Longitude float64 `json:"longitude"`
}

// TakoOutputSettings control card rendering.
type TakoOutputSettings struct {
	ImageDarkMode *bool `json:"imageDarkMode,omitempty"`

	// ForceRefresh applies to instant effort only.
	ForceRefresh *bool `json:"forceRefresh,omitempty"`
}

// BrowserbaseFetch returns the Browserbase fetch tool, named
// "browserbase_fetch", which fetches page content as raw text, Markdown or
// schema-driven JSON. It fails when Schema is not valid JSON.
func BrowserbaseFetch(args BrowserbaseFetchArgs) (model.ProviderTool, error) {
	if len(args.Schema) > 0 && !json.Valid(args.Schema) {
		return model.ProviderTool{}, fmt.Errorf("%w: browserbase fetch schema is not valid JSON", model.ErrInvalidArgument)
	}

	return gatewayTool(browserbaseFetchToolID, "browserbase_fetch", args), nil
}

// BrowserbaseSearch returns the Browserbase search tool, named
// "browserbase_search", which returns structured web results.
func BrowserbaseSearch(args BrowserbaseSearchArgs) model.ProviderTool {
	return gatewayTool(browserbaseSearchToolID, "browserbase_search", args)
}

// ExaSearch returns the Exa search tool, named "exa_search", which returns
// current, token-efficient web excerpts.
func ExaSearch(args ExaSearchArgs) model.ProviderTool {
	return gatewayTool(exaSearchToolID, "exa_search", args)
}

// ParallelSearch returns the Parallel search tool, named "parallel_search",
// which answers a research objective with LLM-optimized excerpts.
func ParallelSearch(args ParallelSearchArgs) model.ProviderTool {
	return gatewayTool(parallelSearchToolID, "parallel_search", args)
}

// PerplexitySearch returns the Perplexity search tool, named
// "perplexity_search", which returns ranked, filtered web results.
func PerplexitySearch(args PerplexitySearchArgs) model.ProviderTool {
	return gatewayTool(perplexitySearchToolID, "perplexity_search", args)
}

// TakoSearch returns the Tako search tool, named "tako_search", which
// searches the web and Tako's knowledge graph for excerpts and structured
// data.
func TakoSearch(args TakoSearchArgs) model.ProviderTool {
	return gatewayTool(takoSearchToolID, "tako_search", args)
}

func gatewayTool[Args any](id string, name string, args Args) model.ProviderTool {
	encoded, err := json.Marshal(args)
	if err != nil {
		// The argument types are plain structs, and raw JSON members are
		// validated before encoding.
		panic(fmt.Sprintf("encode %s tool arguments: %v", id, err))
	}

	return model.ProviderTool{
		ID:   id,
		Name: name,
		Args: encoded,
	}
}
