package mcp

import (
	"encoding/json"
	"fmt"
)

// Implementation identifies an MCP client or server.
type Implementation struct {
	Name    string `json:"name"`
	Version string `json:"version"`
	Title   string `json:"title,omitempty"`
}

// ClientCapabilities are the optional features the client advertises. The
// application handles the corresponding server requests; advertising a
// capability only lets the server send them.
type ClientCapabilities struct {
	Elicitation *ElicitationCapability `json:"elicitation,omitempty"`

	// Extensions advertises protocol extensions by name, such as MCP Apps
	// (see AppClientCapabilities).
	Extensions map[string]json.RawMessage `json:"extensions,omitempty"`

	Experimental map[string]json.RawMessage `json:"experimental,omitempty"`
}

// ElicitationCapability advertises support for elicitation requests.
type ElicitationCapability struct {
	ApplyDefaults *bool `json:"applyDefaults,omitempty"`
}

// ServerCapabilities are the features the server reported. A nil member
// means the server does not support the feature.
type ServerCapabilities struct {
	Experimental map[string]json.RawMessage `json:"experimental,omitempty"`
	Logging      *EmptyCapability           `json:"logging,omitempty"`
	Completions  *EmptyCapability           `json:"completions,omitempty"`
	Prompts      *ListChangedCapability     `json:"prompts,omitempty"`
	Resources    *ResourcesCapability       `json:"resources,omitempty"`
	Tools        *ListChangedCapability     `json:"tools,omitempty"`
	Elicitation  *ElicitationCapability     `json:"elicitation,omitempty"`
	Extensions   map[string]json.RawMessage `json:"extensions,omitempty"`

	// Events advertises experimental event subscriptions.
	Events *ListChangedCapability `json:"events,omitempty"`
}

// EmptyCapability is a capability without settings; its presence enables
// the feature.
type EmptyCapability struct{}

// ListChangedCapability is a capability whose server may notify the client
// when its list changes.
type ListChangedCapability struct {
	ListChanged *bool `json:"listChanged,omitempty"`
}

// ResourcesCapability describes the server's resource support.
type ResourcesCapability struct {
	Subscribe   *bool `json:"subscribe,omitempty"`
	ListChanged *bool `json:"listChanged,omitempty"`
}

// InitializeResult is the server's response to initialize. Persist it with
// the session to reconnect without a new handshake through
// ClientOptions.InitialInitializeResult.
type InitializeResult struct {
	ProtocolVersion string                     `json:"protocolVersion"`
	Capabilities    ServerCapabilities         `json:"capabilities"`
	ServerInfo      Implementation             `json:"serverInfo"`
	Instructions    string                     `json:"instructions,omitempty"`
	Meta            map[string]json.RawMessage `json:"_meta,omitempty"`
}

func (r *InitializeResult) validate(raw json.RawMessage) error {
	return requireFields(raw, "protocolVersion", "capabilities", "serverInfo")
}

// discoverResult is the server's response to server/discover in the
// stateless protocol.
type discoverResult struct {
	SupportedVersions []string                   `json:"supportedVersions"`
	Capabilities      ServerCapabilities         `json:"capabilities"`
	Instructions      string                     `json:"instructions,omitempty"`
	TTLMs             *float64                   `json:"ttlMs,omitempty"`
	CacheScope        string                     `json:"cacheScope,omitempty"`
	Meta              map[string]json.RawMessage `json:"_meta,omitempty"`
}

func (r *discoverResult) validate(raw json.RawMessage) error {
	if err := requireFields(raw, "supportedVersions", "capabilities"); err != nil {
		return err
	}

	if r.CacheScope != "" && r.CacheScope != "public" && r.CacheScope != "private" {
		return fmt.Errorf("invalid cacheScope %q", r.CacheScope)
	}

	return nil
}

// discoveredServerInfo reads the server identity that a server/discover
// result carries in its _meta.
func discoveredServerInfo(meta map[string]json.RawMessage) (Implementation, bool) {
	raw, ok := meta[metaServerInfo]
	if !ok || requireFields(raw, "name", "version") != nil {
		return Implementation{}, false
	}

	var info Implementation
	if err := json.Unmarshal(raw, &info); err != nil {
		return Implementation{}, false
	}

	return info, true
}

// initializeParams are the parameters of initialize.
type initializeParams struct {
	ProtocolVersion string             `json:"protocolVersion"`
	Capabilities    ClientCapabilities `json:"capabilities"`
	ClientInfo      Implementation     `json:"clientInfo"`
}

// PaginatedParams select a page of a list request. The zero value requests
// the first page.
type PaginatedParams struct {
	// Cursor is the NextCursor of the previous page.
	Cursor string                     `json:"cursor,omitempty"`
	Meta   map[string]json.RawMessage `json:"_meta,omitempty"`
}

// Notification is a JSON-RPC notification received from the server, such as
// notifications/tools/list_changed or notifications/progress.
type Notification struct {
	Method string
	Params json.RawMessage
}

// resultValidator checks a decoded result against requirements that JSON
// decoding cannot express, such as required members. raw is the encoded
// result.
type resultValidator interface {
	validate(raw json.RawMessage) error
}

// requireFields reports a missing or null member of the object raw.
func requireFields(raw json.RawMessage, names ...string) error {
	var members map[string]json.RawMessage
	if err := json.Unmarshal(raw, &members); err != nil {
		return err
	}

	for _, name := range names {
		value, ok := members[name]
		if !ok || string(value) == "null" {
			return fmt.Errorf("missing required member %q", name)
		}
	}

	return nil
}
