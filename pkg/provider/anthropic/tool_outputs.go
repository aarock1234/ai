package anthropic

// The types in this file are the results of provider-executed tools, as they
// appear in ToolResult.Result. Results with snake_case keys share their shape
// with the API, so they are also sent back unchanged when the conversation is
// replayed.

// Result type discriminators.
const (
	resultCodeExecution          = "code_execution_result"
	resultEncryptedCodeExecution = "encrypted_code_execution_result"
	resultCodeExecutionError     = "code_execution_tool_result_error"
	resultBashCodeExecution      = "bash_code_execution_result"
	resultBashCodeExecutionError = "bash_code_execution_tool_result_error"
	resultTextEditorError        = "text_editor_code_execution_tool_result_error"
	resultTextEditorView         = "text_editor_code_execution_view_result"
	resultTextEditorCreate       = "text_editor_code_execution_create_result"
	resultTextEditorStrReplace   = "text_editor_code_execution_str_replace_result"
	resultWebFetch               = "web_fetch_result"
	resultWebFetchError          = "web_fetch_tool_result_error"
	resultWebSearch              = "web_search_result"
	resultWebSearchError         = "web_search_tool_result_error"
	resultToolSearch             = "tool_search_tool_search_result"
	resultToolSearchError        = "tool_search_tool_result_error"
	resultAdvisor                = "advisor_result"
	resultAdvisorRedacted        = "advisor_redacted_result"
	resultAdvisorError           = "advisor_tool_result_error"
	resultToolReference          = "tool_reference"
)

// CodeExecutionResult is the result of a code execution run. Type is
// "code_execution_result".
type CodeExecutionResult struct {
	Type       string              `json:"type"`
	Stdout     string              `json:"stdout"`
	Stderr     string              `json:"stderr"`
	ReturnCode int                 `json:"return_code"`
	Content    []CodeExecutionFile `json:"content"`
}

// EncryptedCodeExecutionResult is a code execution result with encrypted
// output, produced by code execution that dynamic web tools provision. Type
// is "encrypted_code_execution_result".
type EncryptedCodeExecutionResult struct {
	Type            string              `json:"type"`
	EncryptedStdout string              `json:"encrypted_stdout"`
	Stderr          string              `json:"stderr"`
	ReturnCode      int                 `json:"return_code"`
	Content         []CodeExecutionFile `json:"content"`
}

// CodeExecutionFile is a file produced by code execution. Type is
// "code_execution_output" or "bash_code_execution_output".
type CodeExecutionFile struct {
	Type   string `json:"type"`
	FileID string `json:"file_id"`
}

// BashCodeExecutionResult is the result of a bash command run by code
// execution. Type is "bash_code_execution_result".
type BashCodeExecutionResult struct {
	Type       string              `json:"type"`
	Stdout     string              `json:"stdout"`
	Stderr     string              `json:"stderr"`
	ReturnCode int                 `json:"return_code"`
	Content    []CodeExecutionFile `json:"content"`
}

// CodeExecutionError is a failed bash or text editor code execution
// subtool. Type is "bash_code_execution_tool_result_error" or
// "text_editor_code_execution_tool_result_error".
type CodeExecutionError struct {
	Type      string `json:"type"`
	ErrorCode string `json:"error_code"`
}

// TextEditorViewResult is the result of viewing a file. Type is
// "text_editor_code_execution_view_result".
type TextEditorViewResult struct {
	Type       string `json:"type"`
	Content    string `json:"content"`
	FileType   string `json:"file_type"`
	NumLines   *int   `json:"num_lines"`
	StartLine  *int   `json:"start_line"`
	TotalLines *int   `json:"total_lines"`
}

// TextEditorCreateResult is the result of creating a file. Type is
// "text_editor_code_execution_create_result".
type TextEditorCreateResult struct {
	Type         string `json:"type"`
	IsFileUpdate bool   `json:"is_file_update"`
}

// TextEditorStrReplaceResult is the result of replacing text in a file. Type
// is "text_editor_code_execution_str_replace_result".
type TextEditorStrReplaceResult struct {
	Type     string   `json:"type"`
	Lines    []string `json:"lines"`
	NewLines *int     `json:"new_lines"`
	NewStart *int     `json:"new_start"`
	OldLines *int     `json:"old_lines"`
	OldStart *int     `json:"old_start"`
}

// ToolResultError is the error result of a web fetch, web search, tool
// search, advisor or code execution tool, such as
// {"type": "web_search_tool_result_error", "errorCode": "unavailable"}.
type ToolResultError struct {
	Type      string `json:"type"`
	ErrorCode string `json:"errorCode"`
}

// WebFetchResult is the result of a web fetch. Type is "web_fetch_result".
type WebFetchResult struct {
	Type        string           `json:"type"`
	URL         string           `json:"url"`
	RetrievedAt *string          `json:"retrievedAt"`
	Content     WebFetchDocument `json:"content"`
}

// WebFetchDocument is a fetched document. Type is "document".
type WebFetchDocument struct {
	Type      string           `json:"type"`
	Title     *string          `json:"title"`
	Citations *CitationsConfig `json:"citations,omitempty"`
	Source    WebFetchSource   `json:"source"`
}

// WebFetchSource is the content of a fetched document: base64 PDF data or
// plain text.
type WebFetchSource struct {
	Type      string `json:"type"`
	MediaType string `json:"mediaType"`
	Data      string `json:"data"`
}

// WebSearchResult is one result of a web search. Type is
// "web_search_result". The web search tool result is a list of them.
type WebSearchResult struct {
	URL              string  `json:"url"`
	Title            *string `json:"title,omitempty"`
	PageAge          *string `json:"pageAge"`
	EncryptedContent string  `json:"encryptedContent"`
	Type             string  `json:"type"`
}

// ToolReference names a tool found by the tool search tool. Type is
// "tool_reference". The tool search result is a list of them.
type ToolReference struct {
	Type     string `json:"type"`
	ToolName string `json:"toolName"`
}

// AdvisorResult is the result of the advisor tool. Type "advisor_result"
// carries Text, "advisor_redacted_result" carries EncryptedContent, which
// must be replayed verbatim, and "advisor_tool_result_error" carries
// ErrorCode. StopReason is "max_tokens" when the advice was truncated.
type AdvisorResult struct {
	Type             string  `json:"type"`
	Text             *string `json:"text,omitempty"`
	EncryptedContent *string `json:"encryptedContent,omitempty"`
	StopReason       *string `json:"stopReason,omitempty"`
	ErrorCode        *string `json:"errorCode,omitempty"`
}
