package googlevertex

import (
	"github.com/aarock1234/ai/pkg/provider/google"
)

// Interactions returns a Gemini Interactions API language model. The
// Interactions API is a location-scoped resource, so requests go to
// .../locations/{location}/interactions. It requires a token source:
// express mode API keys are rejected when a request is made.
func (p *Provider) Interactions(modelID string) *google.InteractionsModel {
	return p.interactionsProvider().Interactions(modelID)
}

// InteractionsAgent returns an Interactions API language model that runs a
// Gemini agent.
func (p *Provider) InteractionsAgent(agent google.InteractionsAgent) *google.InteractionsModel {
	return p.interactionsProvider().InteractionsAgent(agent)
}

// InteractionsManagedAgent returns an Interactions API language model that
// runs a user-defined agent.
func (p *Provider) InteractionsManagedAgent(name string) *google.InteractionsModel {
	return p.interactionsProvider().InteractionsManagedAgent(name)
}

// interactionsProvider returns a Google provider named "google.vertex",
// whose Interactions models report "google.vertex.interactions", with the
// location-scoped base URL.
func (p *Provider) interactionsProvider() *google.Provider {
	s := &p.settings

	options := []google.Option{
		google.WithName(s.name),
		google.WithBaseURLResolver(func() (string, error) {
			if err := s.requireTokenSource("Interactions models"); err != nil {
				return "", err
			}

			return s.resolveBaseURL(true)
		}),
	}

	return google.New(append(options, p.connectionOptions()...)...)
}
