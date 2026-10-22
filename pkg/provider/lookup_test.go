package provider_test

import (
	"testing"

	"github.com/aarock1234/ai/pkg/provider"
	"github.com/aarock1234/ai/pkg/provider/cohere"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

// TestLookupCapabilities verifies capability-based retrieval, including
// rerank-only providers that do not implement the chat composite.
func TestLookupCapabilities(t *testing.T) {
	registry := provider.NewRegistry()
	registry.Register("openai", openai.New())
	registry.Register("cohere", cohere.New())

	if _, ok := registry.Get("openai"); !ok {
		t.Error("Get(openai) = false, want chat provider")
	}
	if _, ok := registry.Get("cohere"); ok {
		t.Error("Get(cohere) = true, want false (rerank-only)")
	}

	if _, ok := provider.Lookup[provider.RerankProvider](registry, "cohere"); !ok {
		t.Error("Lookup[RerankProvider](cohere) = false, want true")
	}
	if _, ok := provider.Lookup[provider.ImageProvider](registry, "openai"); !ok {
		t.Error("Lookup[ImageProvider](openai) = false, want true")
	}
	if _, ok := provider.Lookup[provider.ImageProvider](registry, "cohere"); ok {
		t.Error("Lookup[ImageProvider](cohere) = true, want false")
	}
	if _, ok := provider.Lookup[provider.RerankProvider](registry, "missing"); ok {
		t.Error("Lookup on missing name = true, want false")
	}
}
