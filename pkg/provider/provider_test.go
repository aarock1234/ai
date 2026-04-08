package provider_test

import (
	"testing"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider"
)

// stubProvider is a minimal Provider for testing the registry.
type stubProvider struct {
	name string
}

func (p *stubProvider) Chat(_ string) model.Model               { return nil }
func (p *stubProvider) Embedding(_ string) model.EmbeddingModel { return nil }

func TestRegistryRegisterAndGet(t *testing.T) {
	r := provider.NewRegistry()

	p := &stubProvider{name: "test"}
	r.Register("test", p)

	got, ok := r.Get("test")
	if !ok {
		t.Fatal("Get returned false for registered provider")
	}

	if got != p {
		t.Error("Get returned different provider than registered")
	}
}

func TestRegistryGetMissing(t *testing.T) {
	r := provider.NewRegistry()

	_, ok := r.Get("missing")
	if ok {
		t.Error("Get returned true for unregistered provider")
	}
}

func TestRegistryOverwrite(t *testing.T) {
	r := provider.NewRegistry()

	p1 := &stubProvider{name: "first"}
	p2 := &stubProvider{name: "second"}

	r.Register("test", p1)
	r.Register("test", p2)

	got, ok := r.Get("test")
	if !ok {
		t.Fatal("Get returned false")
	}

	if got != p2 {
		t.Error("Get should return the most recently registered provider")
	}
}

func TestRegistryMultipleProviders(t *testing.T) {
	r := provider.NewRegistry()

	r.Register("openai", &stubProvider{name: "openai"})
	r.Register("openrouter", &stubProvider{name: "openrouter"})

	if _, ok := r.Get("openai"); !ok {
		t.Error("openai not found")
	}

	if _, ok := r.Get("openrouter"); !ok {
		t.Error("openrouter not found")
	}

	if _, ok := r.Get("anthropic"); ok {
		t.Error("anthropic should not be found")
	}
}
