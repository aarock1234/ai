package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strings"
)

// MCP Apps identifiers.
const (
	// AppExtensionName is the capability extension of hosts that render MCP
	// Apps.
	AppExtensionName = "io.modelcontextprotocol/ui"

	// AppMIMEType is the MIME type of HTML resources rendered as MCP Apps.
	AppMIMEType = "text/html;profile=mcp-app"

	// appLegacyResourceURIMetaKey is the deprecated flat _meta key of an app
	// resource URI, still read for older servers.
	appLegacyResourceURIMetaKey = "ui/resourceUri"

	appResourceURIScheme = "ui://"
)

// AppClientCapabilities returns the client capabilities of a host that can
// render MCP Apps, for ClientOptions.Capabilities.
func AppClientCapabilities() ClientCapabilities {
	return ClientCapabilities{
		Extensions: map[string]json.RawMessage{
			AppExtensionName: json.RawMessage(`{"mimeTypes":["` + AppMIMEType + `"]}`),
		},
	}
}

// AppToolVisibility says who may call an app tool.
type AppToolVisibility string

// App tool visibilities.
const (
	AppToolVisibleToModel AppToolVisibility = "model"
	AppToolVisibleToApp   AppToolVisibility = "app"
)

// AppToolMeta is the normalized _meta.ui member of a tool definition.
type AppToolMeta struct {
	// ResourceURI is the ui:// resource that renders the tool.
	ResourceURI string

	// Visibility lists who may call the tool. Nil means the model only.
	Visibility []AppToolVisibility

	// MIMEType is AppMIMEType in tool metadata and empty otherwise.
	MIMEType string

	// Extra holds the other members of _meta.ui.
	Extra map[string]json.RawMessage
}

// MarshalJSON encodes the metadata as one object.
func (m AppToolMeta) MarshalJSON() ([]byte, error) {
	members := make(map[string]any, len(m.Extra)+3)
	for key, value := range m.Extra {
		members[key] = value
	}

	if m.ResourceURI != "" {
		members["resourceUri"] = m.ResourceURI
	}

	if m.Visibility != nil {
		members["visibility"] = m.Visibility
	}

	if m.MIMEType != "" {
		members["mimeType"] = m.MIMEType
	}

	return json.Marshal(members)
}

// appToolMetaOf reads the MCP Apps metadata of a tool. It returns nil for
// tools without it and fails on a resource URI that is not a ui:// string.
func appToolMetaOf(meta map[string]json.RawMessage) (*AppToolMeta, error) {
	uiMeta := decodeObject(meta["ui"])

	resourceURIValue, hasResourceURI := nonNullMember(uiMeta, "resourceUri")
	if !hasResourceURI {
		resourceURIValue, hasResourceURI = nonNullMember(meta, appLegacyResourceURIMetaKey)
	}

	if !hasResourceURI && uiMeta == nil {
		return nil, nil
	}

	appMeta := &AppToolMeta{
		Visibility: decodeVisibility(uiMeta["visibility"]),
		Extra:      maps.Clone(uiMeta),
	}

	delete(appMeta.Extra, "resourceUri")
	delete(appMeta.Extra, "visibility")

	if hasResourceURI {
		var resourceURI string
		if err := json.Unmarshal(resourceURIValue, &resourceURI); err != nil || !strings.HasPrefix(resourceURI, appResourceURIScheme) {
			return nil, fmt.Errorf("invalid mcp app resource uri: %s", resourceURIValue)
		}

		appMeta.ResourceURI = resourceURI
	}

	return appMeta, nil
}

// decodeObject returns the members of raw when it is a JSON object, and nil
// otherwise.
func decodeObject(raw json.RawMessage) map[string]json.RawMessage {
	if !isJSONObject(raw) {
		return nil
	}

	var members map[string]json.RawMessage
	if err := json.Unmarshal(raw, &members); err != nil {
		return nil
	}

	return members
}

func nonNullMember(members map[string]json.RawMessage, name string) (json.RawMessage, bool) {
	value, ok := members[name]
	if !ok || string(value) == "null" {
		return nil, false
	}

	return value, true
}

// decodeVisibility keeps the known visibilities of a JSON array, and returns
// nil for anything else.
func decodeVisibility(raw json.RawMessage) []AppToolVisibility {
	var values []json.RawMessage
	if !isJSONArray(raw) || json.Unmarshal(raw, &values) != nil {
		return nil
	}

	visibility := make([]AppToolVisibility, 0, len(values))
	for _, value := range values {
		var text string
		if json.Unmarshal(value, &text) != nil {
			continue
		}

		if candidate := AppToolVisibility(text); candidate == AppToolVisibleToModel || candidate == AppToolVisibleToApp {
			visibility = append(visibility, candidate)
		}
	}

	return visibility
}

// SplitAppTools splits tool definitions into those the model may call and
// those an app may call. Tools without visibility metadata are visible to
// the model only; a tool may be in both results.
func SplitAppTools(definitions ListToolsResult) (modelVisible ListToolsResult, appVisible ListToolsResult, err error) {
	modelVisible = definitions
	modelVisible.Tools = nil

	appVisible = definitions
	appVisible.Tools = nil

	for _, tool := range definitions.Tools {
		appMeta, err := appToolMetaOf(tool.Meta)
		if err != nil {
			return ListToolsResult{}, ListToolsResult{}, fmt.Errorf("tool %q: %w", tool.Name, err)
		}

		var visibility []AppToolVisibility
		if appMeta != nil {
			visibility = appMeta.Visibility
		}

		if visibility == nil || slices.Contains(visibility, AppToolVisibleToModel) {
			modelVisible.Tools = append(modelVisible.Tools, tool)
		}

		if slices.Contains(visibility, AppToolVisibleToApp) {
			appVisible.Tools = append(appVisible.Tools, tool)
		}
	}

	return modelVisible, appVisible, nil
}

// AppResource is the HTML and rendering metadata of an MCP App.
type AppResource struct {
	URI      string
	MIMEType string
	HTML     string
	Meta     *AppResourceMeta
}

// AppResourceMeta is the host rendering metadata of an app resource, from
// its _meta.ui member. Malformed known members are dropped.
type AppResourceMeta struct {
	PrefersBorder *bool
	CSP           *AppResourceCSP
	Permissions   map[string]json.RawMessage

	// Extra holds the other members.
	Extra map[string]json.RawMessage
}

// MarshalJSON encodes the metadata as one object.
func (m AppResourceMeta) MarshalJSON() ([]byte, error) {
	members := make(map[string]any, len(m.Extra)+3)
	for key, value := range m.Extra {
		members[key] = value
	}

	if m.PrefersBorder != nil {
		members["prefersBorder"] = *m.PrefersBorder
	}

	if m.CSP != nil {
		members["csp"] = m.CSP
	}

	if m.Permissions != nil {
		members["permissions"] = m.Permissions
	}

	return json.Marshal(members)
}

// AppResourceCSP is the content security policy an app resource requests.
// Non-string domains are dropped.
type AppResourceCSP struct {
	ConnectDomains  []string
	ResourceDomains []string
	FrameDomains    []string

	// Extra holds the other members.
	Extra map[string]json.RawMessage
}

// MarshalJSON encodes the policy as one object.
func (p AppResourceCSP) MarshalJSON() ([]byte, error) {
	members := make(map[string]any, len(p.Extra)+3)
	for key, value := range p.Extra {
		members[key] = value
	}

	if p.ConnectDomains != nil {
		members["connectDomains"] = p.ConnectDomains
	}

	if p.ResourceDomains != nil {
		members["resourceDomains"] = p.ResourceDomains
	}

	if p.FrameDomains != nil {
		members["frameDomains"] = p.FrameDomains
	}

	return json.Marshal(members)
}

// ResourceReader reads resources, as Client does.
type ResourceReader interface {
	ReadResource(ctx context.Context, params ReadResourceParams) (*ReadResourceResult, error)
}

// ReadAppResource reads a ui:// resource and normalizes it for rendering.
func ReadAppResource(ctx context.Context, reader ResourceReader, uri string) (*AppResource, error) {
	if !strings.HasPrefix(uri, appResourceURIScheme) {
		return nil, fmt.Errorf("unsupported mcp app resource uri: %s", uri)
	}

	result, err := reader.ReadResource(ctx, ReadResourceParams{
		URI: uri,
	})
	if err != nil {
		return nil, err
	}

	return appResourceFromReadResult(uri, result)
}

func appResourceFromReadResult(uri string, result *ReadResourceResult) (*AppResource, error) {
	for _, contents := range result.Contents {
		if contents.URI != uri {
			continue
		}

		if contents.MIMEType != AppMIMEType {
			return nil, fmt.Errorf("unsupported mcp app resource mime type: %s", contents.MIMEType)
		}

		html := contents.Text
		if contents.IsBlob() {
			html = string(contents.Blob)
		}

		return &AppResource{
			URI:      uri,
			MIMEType: AppMIMEType,
			HTML:     html,
			Meta:     decodeAppResourceMeta(contents.Meta),
		}, nil
	}

	return nil, fmt.Errorf("mcp app resource not found in read result: %s", uri)
}

func decodeAppResourceMeta(meta map[string]json.RawMessage) *AppResourceMeta {
	uiMeta := decodeObject(meta["ui"])
	if uiMeta == nil {
		return nil
	}

	resourceMeta := &AppResourceMeta{
		CSP:         decodeAppResourceCSP(uiMeta["csp"]),
		Permissions: decodeObject(uiMeta["permissions"]),
		Extra:       maps.Clone(uiMeta),
	}

	var prefersBorder bool
	if json.Unmarshal(uiMeta["prefersBorder"], &prefersBorder) == nil && string(uiMeta["prefersBorder"]) != "null" {
		resourceMeta.PrefersBorder = &prefersBorder
	}

	delete(resourceMeta.Extra, "prefersBorder")
	delete(resourceMeta.Extra, "csp")
	delete(resourceMeta.Extra, "permissions")

	return resourceMeta
}

func decodeAppResourceCSP(raw json.RawMessage) *AppResourceCSP {
	members := decodeObject(raw)
	if members == nil {
		return nil
	}

	csp := &AppResourceCSP{
		ConnectDomains:  decodeStrings(members["connectDomains"]),
		ResourceDomains: decodeStrings(members["resourceDomains"]),
		FrameDomains:    decodeStrings(members["frameDomains"]),
		Extra:           maps.Clone(members),
	}

	delete(csp.Extra, "connectDomains")
	delete(csp.Extra, "resourceDomains")
	delete(csp.Extra, "frameDomains")

	return csp
}

// decodeStrings keeps the strings of a JSON array, and returns nil for
// anything else.
func decodeStrings(raw json.RawMessage) []string {
	var values []json.RawMessage
	if !isJSONArray(raw) || json.Unmarshal(raw, &values) != nil {
		return nil
	}

	texts := make([]string, 0, len(values))
	for _, value := range values {
		var text string
		if json.Unmarshal(value, &text) == nil {
			texts = append(texts, text)
		}
	}

	return texts
}
