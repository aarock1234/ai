package fx

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/harness/acp"
)

// profilePath is fx's personal settings file, relative to its private home.
const profilePath = ".fx/settings.json"

// profile is the part of fx's personal settings the adapter controls. fx's
// ACP server reads the startup reasoning effort and fast mode only from
// these settings.
type profile struct {
	Effort   string `json:"effort,omitempty"`
	FastMode bool   `json:"fast_mode,omitempty"`
}

// validateServiceTier rejects service tiers fx does not offer.
func validateServiceTier(tier ServiceTier) error {
	switch tier {
	case "", ServiceTierDefault, ServiceTierFast, ServiceTierUltrafast:
		return nil
	default:
		return fmt.Errorf("fx service tier %q is not supported; supported tiers: %q, %q, %q", tier, ServiceTierDefault, ServiceTierFast, ServiceTierUltrafast)
	}
}

// profileFiles writes fx's personal settings before every launch, so that
// they always reflect the adapter's settings. It is the adapter's
// acp.Settings.AuthenticationFiles; the file holds no credential.
func profileFiles(settings Settings) ([]acp.AuthenticationFile, error) {
	content, err := json.MarshalIndent(profile{
		Effort:   settings.Effort,
		FastMode: settings.ServiceTier == ServiceTierFast,
	}, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("encode fx settings: %w", err)
	}

	return []acp.AuthenticationFile{
		{
			Path:    profilePath,
			Content: string(content) + "\n",
		},
	}, nil
}
