// Package toolnames maps between application tool names and the fixed names
// providers use for their built-in tools.
package toolnames

import "github.com/aarock1234/ai/pkg/model"

// Mapping translates tool names in both directions. Names without a mapping
// pass through unchanged.
type Mapping struct {
	toProvider    map[string]string
	toApplication map[string]string
}

// New builds a mapping from the call's tools. providerNames maps a provider
// tool ID, such as "anthropic.web_search_20250305", to the name the provider
// uses on the wire, such as "web_search".
func New(tools []model.Tool, providerNames map[string]string) Mapping {
	mapping := Mapping{
		toProvider:    make(map[string]string),
		toApplication: make(map[string]string),
	}

	for _, tool := range tools {
		providerTool, ok := tool.(model.ProviderTool)
		if !ok {
			continue
		}

		providerName, known := providerNames[providerTool.ID]
		if !known {
			continue
		}

		mapping.toProvider[providerTool.Name] = providerName
		mapping.toApplication[providerName] = providerTool.Name
	}

	return mapping
}

// ToProvider returns the wire name for an application tool name.
func (m Mapping) ToProvider(applicationName string) string {
	if name, ok := m.toProvider[applicationName]; ok {
		return name
	}

	return applicationName
}

// ToApplication returns the application name for a wire tool name.
func (m Mapping) ToApplication(providerName string) string {
	if name, ok := m.toApplication[providerName]; ok {
		return name
	}

	return providerName
}
