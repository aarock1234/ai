package middleware

// identifiedModel is the identity that every model type reports.
type identifiedModel interface {
	Provider() string
	ModelID() string
}

// modelIdentity resolves the identity of a wrapped model. The IDs set when
// wrapping take precedence over the middleware overrides, which take
// precedence over the base model.
type modelIdentity[Model identifiedModel] struct {
	base Model

	providerID string
	modelID    string

	overrideProvider func(base Model) string
	overrideModelID  func(base Model) string
}

// Provider returns the provider name of the wrapped model.
func (i modelIdentity[Model]) Provider() string {
	switch {
	case i.providerID != "":
		return i.providerID
	case i.overrideProvider != nil:
		return i.overrideProvider(i.base)
	default:
		return i.base.Provider()
	}
}

// ModelID returns the model ID of the wrapped model.
func (i modelIdentity[Model]) ModelID() string {
	switch {
	case i.modelID != "":
		return i.modelID
	case i.overrideModelID != nil:
		return i.overrideModelID(i.base)
	default:
		return i.base.ModelID()
	}
}
