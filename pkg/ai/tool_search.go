package ai

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
)

const (
	defaultToolSearchName       = "tool_search"
	defaultToolSearchMaxResults = 5
)

// errToolSearchUnbound reports a tool search executed outside a generation,
// which binds it to the generation's deferred tools.
var errToolSearchUnbound = errors.New("tool search must be bound by a generation")

// toolSearchInputSchema is the input schema of a tool search.
var toolSearchInputSchema = json.RawMessage(`{
	"type": "object",
	"properties": {
		"query": {
			"type": "string",
			"minLength": 1
		}
	},
	"required": ["query"],
	"additionalProperties": false
}`)

// toolSearchOutputSchema is the output schema of a tool search.
var toolSearchOutputSchema = json.RawMessage(`{
	"type": "object",
	"properties": {
		"tools": {
			"type": "array",
			"items": {
				"type": "object",
				"properties": {
					"name": {
						"type": "string"
					},
					"description": {
						"type": "string"
					}
				},
				"required": ["name"],
				"additionalProperties": false
			}
		}
	},
	"required": ["tools"],
	"additionalProperties": false
}`)

// camelCaseBoundary and searchTerm split names and descriptions into search
// terms, as upstream's tokenizer does.
var (
	camelCaseBoundary = regexp.MustCompile(`([a-z\d])([A-Z])`)
	searchTerm        = regexp.MustCompile(`[\p{L}\p{N}]+`)
)

// ToolSearchOptions configure ToolSearch.
type ToolSearchOptions struct {
	// Name is the name of the search tool. Empty selects "tool_search".
	Name string

	// Search ranks the eligible deferred tools for a query and returns their
	// names, best first. Unknown names and duplicates are ignored before
	// the result limit applies. Nil matches the keywords of the query
	// against tool names and descriptions.
	Search ToolSearchFunc

	// MaxResults limits the tools returned per search. Zero selects 5.
	MaxResults int
}

// ToolSearchFunc ranks the tools of a search request and returns their
// names, best first.
type ToolSearchFunc func(ctx context.Context, request ToolSearchRequest) ([]string, error)

// ToolSearchRequest is one search for deferred tools.
type ToolSearchRequest struct {
	Query string

	// Tools are the deferred tools the search may return.
	Tools []ToolSearchMatch
}

// ToolSearchMatch describes a tool that a search may return.
type ToolSearchMatch struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
}

// ToolSearchInput is the input of a tool search call.
type ToolSearchInput struct {
	Query string `json:"query"`
}

// ToolSearchOutput is the output of a tool search call.
type ToolSearchOutput struct {
	Tools []ToolSearchMatch `json:"tools"`
}

// toolSearchConfig marks a tool as a tool search and holds its settings.
type toolSearchConfig struct {
	search     ToolSearchFunc
	maxResults int
}

// ToolSearch returns a tool that searches the deferred tools of a
// generation, those with DeferLoading set, by name and description. The
// tools it finds are declared to the model from the next step on. A
// generation hides deferred tools until a search discovers them.
func ToolSearch(options ToolSearchOptions) Tool {
	maxResults := cmp.Or(options.MaxResults, defaultToolSearchMaxResults)

	tool := Tool{
		Name:         cmp.Or(options.Name, defaultToolSearchName),
		Description:  toolSearchDescription(options.Search != nil, maxResults),
		InputSchema:  toolSearchInputSchema,
		OutputSchema: toolSearchOutputSchema,
		Execute: func(context.Context, json.RawMessage, ToolExecutionOptions) (any, error) {
			return nil, errToolSearchUnbound
		},
		search: &toolSearchConfig{
			search:     options.Search,
			maxResults: maxResults,
		},
	}

	if maxResults < 1 {
		tool.err = &InvalidArgumentError{
			Parameter: "maxResults",
			Value:     options.MaxResults,
			Message:   "maxResults must be a positive integer",
		}
	}

	return tool
}

func toolSearchDescription(customSearch bool, maxResults int) string {
	purpose := "Search for tools by keywords in their names and descriptions."
	if customSearch {
		purpose = "Search for tools matching a query."
	}

	limit := strconv.Itoa(maxResults)
	if maxResults == defaultToolSearchMaxResults {
		limit = "five"
	}

	return fmt.Sprintf(
		"%s Returns up to %s matching tools. Matches become available on the next model step, after this execution finishes. "+
			"Wait for their tool definitions before calling the discovered tools. If no tools match, try different keywords.",
		purpose,
		limit,
	)
}

// isToolSearch reports whether a tool was created by ToolSearch.
func (t Tool) isToolSearch() bool {
	return t.search != nil
}

// toolSearchState tracks the tools discovered in one generation. Searches
// run on tool goroutines, so the discovered set is guarded by a mutex.
type toolSearchState struct {
	enabled bool
	callers resolvedToolCallers

	mu         sync.Mutex
	discovered map[string]bool
}

// newToolSearchState validates the deferred tools and search tools of a
// generation. Each must be callable directly or through a local caller that
// prepares a model message, and a search tool must not defer loading.
func newToolSearchState(tools []Tool, callers resolvedToolCallers) (*toolSearchState, error) {
	state := &toolSearchState{
		callers:    callers,
		discovered: make(map[string]bool),
	}

	for _, tool := range tools {
		if !tool.DeferLoading && !tool.isToolSearch() {
			continue
		}

		state.enabled = true

		if err := state.validateDiscoverableTool(tools, tool); err != nil {
			return nil, err
		}
	}

	return state, nil
}

// validateDiscoverableTool checks a deferred tool or a search tool: a
// search tool must not defer loading, and every caller of the tool must
// keep it visible to the model.
func (s *toolSearchState) validateDiscoverableTool(tools []Tool, tool Tool) error {
	if tool.isToolSearch() && tool.DeferLoading {
		return &InvalidArgumentError{
			Parameter: "tools",
			Value:     tool.Name,
			Message:   fmt.Sprintf("tool search %q must not defer loading", tool.Name),
		}
	}

	for _, callerName := range s.callers.callersOf(tool.Name) {
		if callerName == DirectToolCall || preparesModelMessage(tools, callerName) {
			continue
		}

		return &InvalidArgumentError{
			Parameter: "tools",
			Value:     tool.Name,
			Message: fmt.Sprintf(
				"tool %q must be callable directly or through a local tool caller that prepares a model message, not through %q",
				tool.Name,
				callerName,
			),
		}
	}

	return nil
}

// preparesModelMessage reports whether the named tool is a local caller
// with a model message, which keeps deferred tools discoverable.
func preparesModelMessage(tools []Tool, callerName string) bool {
	callerTool, found := findTool(tools, callerName)
	if !found {
		return false
	}

	caller, isLocal := callerTool.Caller.(LocalToolCaller)

	return isLocal && caller.PrepareModelMessage != nil
}

// prepare hides the deferred tools that no search has discovered and binds
// the search tools to the deferred tools they may return. Tools discovered
// during the step become available in the next step.
func (s *toolSearchState) prepare(active []Tool) []Tool {
	if !s.enabled {
		return active
	}

	discovered := s.discoveredSnapshot()

	prepared := make([]Tool, 0, len(active))

	for _, tool := range active {
		if tool.DeferLoading && !discovered[tool.Name] {
			continue
		}

		if tool.isToolSearch() {
			tool = s.bind(tool, active)
		}

		prepared = append(prepared, tool)
	}

	return prepared
}

func (s *toolSearchState) discoveredSnapshot() map[string]bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	snapshot := make(map[string]bool, len(s.discovered))
	for name := range s.discovered {
		snapshot[name] = true
	}

	return snapshot
}

func (s *toolSearchState) markDiscovered(matches []ToolSearchMatch) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, match := range matches {
		s.discovered[match.Name] = true
	}
}

// bind returns the search tool with an Execute function that searches the
// deferred tools sharing one of its callers.
func (s *toolSearchState) bind(searchTool Tool, active []Tool) Tool {
	candidates := s.searchCandidates(searchTool, active)
	config := *searchTool.search

	searchTool.Execute = func(ctx context.Context, input json.RawMessage, _ ToolExecutionOptions) (any, error) {
		var request ToolSearchInput
		if err := json.Unmarshal(input, &request); err != nil {
			return nil, fmt.Errorf("decode tool search input: %w", err)
		}

		matches, err := searchTools(ctx, toolSearch{
			config:     config,
			query:      request.Query,
			candidates: candidates,
		})
		if err != nil {
			return nil, err
		}

		s.markDiscovered(matches)

		return ToolSearchOutput{
			Tools: matches,
		}, nil
	}

	return searchTool
}

// searchCandidates returns the deferred tools that share a caller with the
// search tool. Callers that are not active in the step are ignored.
func (s *toolSearchState) searchCandidates(searchTool Tool, active []Tool) []ToolSearchMatch {
	searchCallers := s.activeCallersOf(searchTool.Name, active)

	candidates := make([]ToolSearchMatch, 0, len(active))

	for _, tool := range active {
		if !tool.DeferLoading || tool.isToolSearch() {
			continue
		}

		toolCallers := s.callers.callersOf(tool.Name)
		sharesCaller := slices.ContainsFunc(searchCallers, func(callerName string) bool {
			return slices.Contains(toolCallers, callerName)
		})

		if sharesCaller {
			candidates = append(candidates, ToolSearchMatch{
				Name:        tool.Name,
				Description: tool.Description,
			})
		}
	}

	return candidates
}

// activeCallersOf returns the callers of a tool that can call it in the
// step: DirectToolCall and the caller tools that are active.
func (s *toolSearchState) activeCallersOf(toolName string, active []Tool) []string {
	var callers []string

	for _, callerName := range s.callers.callersOf(toolName) {
		_, isActive := findTool(active, callerName)

		if callerName == DirectToolCall || isActive {
			callers = append(callers, callerName)
		}
	}

	return callers
}

// toolSearch is one execution of a bound search tool.
type toolSearch struct {
	config     toolSearchConfig
	query      string
	candidates []ToolSearchMatch
}

// searchTools ranks the candidates and returns at most maxResults known,
// distinct matches in ranked order.
func searchTools(ctx context.Context, search toolSearch) ([]ToolSearchMatch, error) {
	ranked, err := rankToolSearchCandidates(ctx, search)
	if err != nil {
		return nil, err
	}

	matches := make([]ToolSearchMatch, 0, search.config.maxResults)
	seen := make(map[string]bool, len(ranked))

	for _, name := range ranked {
		if len(matches) == search.config.maxResults {
			break
		}

		index := slices.IndexFunc(search.candidates, func(candidate ToolSearchMatch) bool {
			return candidate.Name == name
		})
		if index < 0 || seen[name] {
			continue
		}

		seen[name] = true
		matches = append(matches, search.candidates[index])
	}

	return matches, nil
}

func rankToolSearchCandidates(ctx context.Context, search toolSearch) ([]string, error) {
	if search.config.search != nil {
		ranked, err := search.config.search(ctx, ToolSearchRequest{
			Query: search.query,
			Tools: slices.Clone(search.candidates),
		})
		if err != nil {
			return nil, fmt.Errorf("search tools: %w", err)
		}

		return ranked, nil
	}

	return rankByKeywords(search.query, search.candidates), nil
}

// scoredTool is a candidate with its keyword score.
type scoredTool struct {
	name  string
	score int
}

// rankByKeywords scores each candidate with two points per query term in
// its name and one per term in its description, and returns the matching
// names by descending score. Ties keep declaration order.
func rankByKeywords(query string, candidates []ToolSearchMatch) []string {
	terms := slices.Compact(slices.Sorted(slices.Values(tokenizeSearchText(query))))

	var scored []scoredTool

	for _, candidate := range candidates {
		nameTerms := tokenizeSearchText(candidate.Name)
		descriptionTerms := tokenizeSearchText(candidate.Description)

		score := 0
		for _, term := range terms {
			if slices.Contains(nameTerms, term) {
				score += 2
			}

			if slices.Contains(descriptionTerms, term) {
				score++
			}
		}

		if score > 0 {
			scored = append(scored, scoredTool{
				name:  candidate.Name,
				score: score,
			})
		}
	}

	slices.SortStableFunc(scored, func(first, second scoredTool) int {
		return cmp.Compare(second.score, first.score)
	})

	names := make([]string, 0, len(scored))
	for _, tool := range scored {
		names = append(names, tool.name)
	}

	return names
}

// tokenizeSearchText splits camel case, lowercases and returns the runs of
// letters and digits.
func tokenizeSearchText(text string) []string {
	separated := camelCaseBoundary.ReplaceAllString(text, "$1 $2")

	return searchTerm.FindAllString(strings.ToLower(separated), -1)
}
