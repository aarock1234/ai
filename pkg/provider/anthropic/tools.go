package anthropic

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// Provider tool IDs.
const (
	toolAdvisor20260301         = "anthropic.advisor_20260301"
	toolBash20241022            = "anthropic.bash_20241022"
	toolBash20250124            = "anthropic.bash_20250124"
	toolCodeExecution20250522   = "anthropic.code_execution_20250522"
	toolCodeExecution20250825   = "anthropic.code_execution_20250825"
	toolCodeExecution20260120   = "anthropic.code_execution_20260120"
	toolComputer20241022        = "anthropic.computer_20241022"
	toolComputer20250124        = "anthropic.computer_20250124"
	toolComputer20251124        = "anthropic.computer_20251124"
	toolComputerToolset20260801 = "anthropic.computer_toolset_20260801"
	toolMemory20250818          = "anthropic.memory_20250818"
	toolTextEditor20241022      = "anthropic.text_editor_20241022"
	toolTextEditor20250124      = "anthropic.text_editor_20250124"
	toolTextEditor20250429      = "anthropic.text_editor_20250429"
	toolTextEditor20250728      = "anthropic.text_editor_20250728"
	toolSearchRegex20251119     = "anthropic.tool_search_regex_20251119"
	toolSearchBM2520251119      = "anthropic.tool_search_bm25_20251119"
	toolWebFetch20250910        = "anthropic.web_fetch_20250910"
	toolWebFetch20260209        = "anthropic.web_fetch_20260209"
	toolWebFetch20260318        = "anthropic.web_fetch_20260318"
	toolWebSearch20250305       = "anthropic.web_search_20250305"
	toolWebSearch20260209       = "anthropic.web_search_20260209"
	toolWebSearch20260318       = "anthropic.web_search_20260318"
)

// Tool names the API uses for provider tools.
const (
	nameAdvisor                 = "advisor"
	nameBash                    = "bash"
	nameCodeExecution           = "code_execution"
	nameComputer                = "computer"
	nameMemory                  = "memory"
	nameStrReplaceEditor        = "str_replace_editor"
	nameStrReplaceBased         = "str_replace_based_edit_tool"
	nameToolSearchRegex         = "tool_search_tool_regex"
	nameToolSearchBM25          = "tool_search_tool_bm25"
	nameWebFetch                = "web_fetch"
	nameWebSearch               = "web_search"
	nameBashCodeExecution       = "bash_code_execution"
	nameTextEditorCodeExecution = "text_editor_code_execution"
)

// providerToolNames maps provider tool IDs to the names the API uses for
// them.
var providerToolNames = map[string]string{
	toolCodeExecution20250522:   nameCodeExecution,
	toolCodeExecution20250825:   nameCodeExecution,
	toolCodeExecution20260120:   nameCodeExecution,
	toolComputer20241022:        nameComputer,
	toolComputer20250124:        nameComputer,
	toolComputer20251124:        nameComputer,
	toolComputerToolset20260801: nameComputer,
	toolTextEditor20241022:      nameStrReplaceEditor,
	toolTextEditor20250124:      nameStrReplaceEditor,
	toolTextEditor20250429:      nameStrReplaceBased,
	toolTextEditor20250728:      nameStrReplaceBased,
	toolBash20241022:            nameBash,
	toolBash20250124:            nameBash,
	toolMemory20250818:          nameMemory,
	toolWebSearch20250305:       nameWebSearch,
	toolWebSearch20260209:       nameWebSearch,
	toolWebSearch20260318:       nameWebSearch,
	toolWebFetch20250910:        nameWebFetch,
	toolWebFetch20260209:        nameWebFetch,
	toolWebFetch20260318:        nameWebFetch,
	toolSearchRegex20251119:     nameToolSearchRegex,
	toolSearchBM2520251119:      nameToolSearchBM25,
	toolAdvisor20260301:         nameAdvisor,
}

// AdvisorArgs configure the advisor tool, which consults a stronger model
// during generation.
type AdvisorArgs struct {
	// Model is the advisor model ID. It must be at least as capable as the
	// executor model.
	Model string `json:"model"`

	// MaxUses caps advisor calls per request.
	MaxUses *int `json:"maxUses,omitempty"`

	// MaxTokens caps the tokens of each advisor call. It is at least 1024.
	MaxTokens *int `json:"maxTokens,omitempty"`

	// Caching enables prompt caching of the advisor's own transcript.
	Caching *AdvisorCaching `json:"caching,omitempty"`
}

// AdvisorCaching enables advisor transcript caching. Type must be
// CacheEphemeral.
type AdvisorCaching struct {
	Type CacheControlType `json:"type"`
	TTL  CacheTTL         `json:"ttl"`
}

// ComputerArgs configure the computer use tools before version 20251124.
type ComputerArgs struct {
	DisplayWidthPx  int `json:"displayWidthPx"`
	DisplayHeightPx int `json:"displayHeightPx"`

	// DisplayNumber selects an X11 display.
	DisplayNumber *int `json:"displayNumber,omitempty"`
}

// Computer20251124Args configure the computer use tool version 20251124.
type Computer20251124Args struct {
	DisplayWidthPx  int  `json:"displayWidthPx"`
	DisplayHeightPx int  `json:"displayHeightPx"`
	DisplayNumber   *int `json:"displayNumber,omitempty"`

	// EnableZoom enables the zoom action.
	EnableZoom *bool `json:"enableZoom,omitempty"`
}

// ComputerAction is a member of the computer toolset.
type ComputerAction string

// Computer toolset members.
const (
	ComputerScreenshot     ComputerAction = "screenshot"
	ComputerZoom           ComputerAction = "zoom"
	ComputerLeftClick      ComputerAction = "left_click"
	ComputerRightClick     ComputerAction = "right_click"
	ComputerMiddleClick    ComputerAction = "middle_click"
	ComputerDoubleClick    ComputerAction = "double_click"
	ComputerTripleClick    ComputerAction = "triple_click"
	ComputerLeftClickDrag  ComputerAction = "left_click_drag"
	ComputerMouseMove      ComputerAction = "mouse_move"
	ComputerLeftMouseDown  ComputerAction = "left_mouse_down"
	ComputerLeftMouseUp    ComputerAction = "left_mouse_up"
	ComputerCursorPosition ComputerAction = "cursor_position"
	ComputerScroll         ComputerAction = "scroll"
	ComputerType           ComputerAction = "type"
	ComputerKey            ComputerAction = "key"
	ComputerHoldKey        ComputerAction = "hold_key"
	ComputerWait           ComputerAction = "wait"
)

var computerActions = []ComputerAction{
	ComputerScreenshot,
	ComputerZoom,
	ComputerLeftClick,
	ComputerRightClick,
	ComputerMiddleClick,
	ComputerDoubleClick,
	ComputerTripleClick,
	ComputerLeftClickDrag,
	ComputerMouseMove,
	ComputerLeftMouseDown,
	ComputerLeftMouseUp,
	ComputerCursorPosition,
	ComputerScroll,
	ComputerType,
	ComputerKey,
	ComputerHoldKey,
	ComputerWait,
}

// ComputerToolsetArgs configure the computer toolset. Member calls arrive as
// calls of the toolset tool, with the member name in the input's "action".
type ComputerToolsetArgs struct {
	Configs map[ComputerAction]ComputerMemberConfig `json:"configs,omitempty"`
}

// ComputerMemberConfig enables, disables or defers one toolset member.
type ComputerMemberConfig struct {
	Enabled      *bool `json:"enabled,omitempty"`
	DeferLoading *bool `json:"deferLoading,omitempty"`
}

// TextEditor20250728Args configure the text editor tool version 20250728.
type TextEditor20250728Args struct {
	// MaxCharacters truncates viewed files.
	MaxCharacters *int `json:"maxCharacters,omitempty"`
}

// WebFetchArgs configure the web fetch tools before version 20260318.
type WebFetchArgs struct {
	MaxUses          *int             `json:"maxUses,omitempty"`
	AllowedDomains   []string         `json:"allowedDomains,omitempty"`
	BlockedDomains   []string         `json:"blockedDomains,omitempty"`
	Citations        *CitationsConfig `json:"citations,omitempty"`
	MaxContentTokens *int             `json:"maxContentTokens,omitempty"`
}

// ResponseInclusion controls whether fetched or searched content is included
// in the response.
type ResponseInclusion string

// Response inclusion modes.
const (
	ResponseInclusionFull     ResponseInclusion = "full"
	ResponseInclusionExcluded ResponseInclusion = "excluded"
)

// WebFetch20260318Args configure the web fetch tool version 20260318.
type WebFetch20260318Args struct {
	MaxUses           *int              `json:"maxUses,omitempty"`
	AllowedDomains    []string          `json:"allowedDomains,omitempty"`
	BlockedDomains    []string          `json:"blockedDomains,omitempty"`
	Citations         *CitationsConfig  `json:"citations,omitempty"`
	MaxContentTokens  *int              `json:"maxContentTokens,omitempty"`
	UseCache          *bool             `json:"useCache,omitempty"`
	ResponseInclusion ResponseInclusion `json:"responseInclusion,omitempty"`
}

// WebSearchArgs configure the web search tools before version 20260318.
type WebSearchArgs struct {
	MaxUses        *int          `json:"maxUses,omitempty"`
	AllowedDomains []string      `json:"allowedDomains,omitempty"`
	BlockedDomains []string      `json:"blockedDomains,omitempty"`
	UserLocation   *UserLocation `json:"userLocation,omitempty"`
}

// WebSearch20260318Args configure the web search tool version 20260318.
type WebSearch20260318Args struct {
	MaxUses           *int              `json:"maxUses,omitempty"`
	AllowedDomains    []string          `json:"allowedDomains,omitempty"`
	BlockedDomains    []string          `json:"blockedDomains,omitempty"`
	UserLocation      *UserLocation     `json:"userLocation,omitempty"`
	ResponseInclusion ResponseInclusion `json:"responseInclusion,omitempty"`
}

// UserLocation localizes search results. Type must be "approximate".
type UserLocation struct {
	Type     string `json:"type"`
	City     string `json:"city,omitempty"`
	Region   string `json:"region,omitempty"`
	Country  string `json:"country,omitempty"`
	Timezone string `json:"timezone,omitempty"`
}

// Advisor20260301 returns the advisor tool, which the provider executes.
func Advisor20260301(args AdvisorArgs) model.ProviderTool {
	return newProviderTool(toolAdvisor20260301, nameAdvisor, args)
}

// Bash20241022 returns the bash tool version 20241022, which the
// application executes.
func Bash20241022() model.ProviderTool {
	return newProviderTool(toolBash20241022, nameBash, struct{}{})
}

// Bash20250124 returns the bash tool version 20250124, which the
// application executes.
func Bash20250124() model.ProviderTool {
	return newProviderTool(toolBash20250124, nameBash, struct{}{})
}

// CodeExecution20250522 returns the code execution tool version 20250522,
// which the provider executes.
func CodeExecution20250522() model.ProviderTool {
	return newProviderTool(toolCodeExecution20250522, nameCodeExecution, struct{}{})
}

// CodeExecution20250825 returns the code execution tool version 20250825,
// which the provider executes with bash and text editor subtools.
func CodeExecution20250825() model.ProviderTool {
	return newProviderTool(toolCodeExecution20250825, nameCodeExecution, struct{}{})
}

// CodeExecution20260120 returns the code execution tool version 20260120,
// which the provider executes.
func CodeExecution20260120() model.ProviderTool {
	return newProviderTool(toolCodeExecution20260120, nameCodeExecution, struct{}{})
}

// Computer20241022 returns the computer use tool version 20241022, which the
// application executes.
func Computer20241022(args ComputerArgs) model.ProviderTool {
	return newProviderTool(toolComputer20241022, nameComputer, args)
}

// Computer20250124 returns the computer use tool version 20250124, which the
// application executes.
func Computer20250124(args ComputerArgs) model.ProviderTool {
	return newProviderTool(toolComputer20250124, nameComputer, args)
}

// Computer20251124 returns the computer use tool version 20251124, which the
// application executes.
func Computer20251124(args Computer20251124Args) model.ProviderTool {
	return newProviderTool(toolComputer20251124, nameComputer, args)
}

// ComputerToolset20260801 returns the computer toolset, which the
// application executes.
func ComputerToolset20260801(args ComputerToolsetArgs) model.ProviderTool {
	return newProviderTool(toolComputerToolset20260801, nameComputer, args)
}

// Memory20250818 returns the memory tool, which the application executes.
func Memory20250818() model.ProviderTool {
	return newProviderTool(toolMemory20250818, nameMemory, struct{}{})
}

// TextEditor20241022 returns the text editor tool version 20241022, which the
// application executes.
func TextEditor20241022() model.ProviderTool {
	return newProviderTool(toolTextEditor20241022, nameStrReplaceEditor, struct{}{})
}

// TextEditor20250124 returns the text editor tool version 20250124, which the
// application executes.
func TextEditor20250124() model.ProviderTool {
	return newProviderTool(toolTextEditor20250124, nameStrReplaceEditor, struct{}{})
}

// TextEditor20250429 returns the text editor tool version 20250429, which the
// application executes.
func TextEditor20250429() model.ProviderTool {
	return newProviderTool(toolTextEditor20250429, nameStrReplaceBased, struct{}{})
}

// TextEditor20250728 returns the text editor tool version 20250728, which the
// application executes.
func TextEditor20250728(args TextEditor20250728Args) model.ProviderTool {
	return newProviderTool(toolTextEditor20250728, nameStrReplaceBased, args)
}

// ToolSearchRegex20251119 returns the regex tool search tool, which the
// provider executes to discover deferred tools.
func ToolSearchRegex20251119() model.ProviderTool {
	return newProviderTool(toolSearchRegex20251119, nameToolSearchRegex, struct{}{})
}

// ToolSearchBM25V20251119 returns the BM25 tool search tool version
// 20251119, which the provider executes to discover deferred tools.
func ToolSearchBM25V20251119() model.ProviderTool {
	return newProviderTool(toolSearchBM2520251119, nameToolSearchBM25, struct{}{})
}

// WebFetch20250910 returns the web fetch tool version 20250910, which the
// provider executes.
func WebFetch20250910(args WebFetchArgs) model.ProviderTool {
	return newProviderTool(toolWebFetch20250910, nameWebFetch, args)
}

// WebFetch20260209 returns the web fetch tool version 20260209, which the
// provider executes with dynamic filtering.
func WebFetch20260209(args WebFetchArgs) model.ProviderTool {
	return newProviderTool(toolWebFetch20260209, nameWebFetch, args)
}

// WebFetch20260318 returns the web fetch tool version 20260318, which the
// provider executes.
func WebFetch20260318(args WebFetch20260318Args) model.ProviderTool {
	return newProviderTool(toolWebFetch20260318, nameWebFetch, args)
}

// WebSearch20250305 returns the web search tool version 20250305, which the
// provider executes.
func WebSearch20250305(args WebSearchArgs) model.ProviderTool {
	return newProviderTool(toolWebSearch20250305, nameWebSearch, args)
}

// WebSearch20260209 returns the web search tool version 20260209, which the
// provider executes with dynamic filtering.
func WebSearch20260209(args WebSearchArgs) model.ProviderTool {
	return newProviderTool(toolWebSearch20260209, nameWebSearch, args)
}

// WebSearch20260318 returns the web search tool version 20260318, which the
// provider executes.
func WebSearch20260318(args WebSearch20260318Args) model.ProviderTool {
	return newProviderTool(toolWebSearch20260318, nameWebSearch, args)
}

func newProviderTool[Args any](id string, name string, args Args) model.ProviderTool {
	return model.ProviderTool{
		ID:   id,
		Name: name,
		Args: mustEncodeJSON(args),
	}
}

// decodeToolArgs decodes the arguments of a provider tool into their typed
// form. Absent arguments decode to the zero value.
func decodeToolArgs[Args any](tool model.ProviderTool) (Args, error) {
	var args Args
	if len(tool.Args) == 0 {
		return args, nil
	}

	if err := json.Unmarshal(tool.Args, &args); err != nil {
		return args, fmt.Errorf("%w: arguments of tool %s: %w", model.ErrInvalidArgument, tool.ID, err)
	}

	return args, nil
}

// mustEncodeJSON encodes plain data or JSON already decoded from the API.
// Neither can fail to encode, so a failure is an internal invariant
// violation.
func mustEncodeJSON[T any](value T) json.RawMessage {
	encoded, err := json.Marshal(value)
	if err != nil {
		panic(fmt.Sprintf("anthropic: encode %T: %v", value, err))
	}

	return encoded
}
