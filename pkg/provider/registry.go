package provider

import "sync"

// Registry maps provider names to Provider instances for dynamic,
// config-driven provider selection. It is safe for concurrent use.
type Registry struct {
	mu        sync.RWMutex
	providers map[string]Provider
}

// NewRegistry creates an empty provider registry.
func NewRegistry() *Registry {
	return &Registry{
		providers: make(map[string]Provider),
	}
}

// Register associates a provider name with a Provider implementation.
// If a provider with the same name is already registered, it is
// replaced.
func (r *Registry) Register(name string, p Provider) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.providers[name] = p
}

// Get retrieves a provider by name. The second return value reports
// whether the provider was found.
func (r *Registry) Get(name string) (Provider, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	p, ok := r.providers[name]

	return p, ok
}
