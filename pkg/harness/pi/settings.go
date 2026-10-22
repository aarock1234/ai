package pi

import (
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"path"
	"regexp"
	"slices"
)

// extensionNamePattern accepts plain extension file names.
var extensionNamePattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*\.(ts|js)$`)

// codemodeExposures are the MCP exposures that need Pi's codemode extension,
// which the adapter does not load.
var codemodeExposures = []string{
	"codemode",
	"codemode-deferred",
}

// validateSettings checks the settings New accepts.
func validateSettings(settings Settings) error {
	validators := []func(settings Settings) error{
		validateThinkingLevel,
		validateCacheRetention,
		validateExtensions,
		validatePathPolicy,
		validateMCPServers,
		validateProviders,
	}

	for _, validate := range validators {
		if err := validate(settings); err != nil {
			return err
		}
	}

	return nil
}

// validateThinkingLevel accepts Pi's thinking levels.
func validateThinkingLevel(settings Settings) error {
	levels := []ThinkingLevel{
		"",
		ThinkingOff,
		ThinkingMinimal,
		ThinkingLow,
		ThinkingMedium,
		ThinkingHigh,
		ThinkingXHigh,
		ThinkingMax,
	}

	if !slices.Contains(levels, settings.ThinkingLevel) {
		return fmt.Errorf("pi thinking level %q is not supported", settings.ThinkingLevel)
	}

	return nil
}

// validateCacheRetention accepts the retentions Pi's command line can
// select. Pi's RPC mode has no request-level retention option, so "none"
// cannot be honored.
func validateCacheRetention(settings Settings) error {
	switch settings.CacheRetention {
	case "", CacheRetentionShort, CacheRetentionLong:
		return nil
	case CacheRetentionNone:
		return errors.New("pi cache retention \"none\" is not supported: Pi's RPC mode can only select the short or long retention")
	default:
		return fmt.Errorf("pi cache retention %q is not supported", settings.CacheRetention)
	}
}

// validateExtensions requires unique, plain extension file names.
func validateExtensions(settings Settings) error {
	seen := make(map[string]bool, len(settings.Extensions))

	for _, extension := range settings.Extensions {
		if !extensionNamePattern.MatchString(extension.Name) {
			return fmt.Errorf("pi extension name %q must be a plain .ts or .js file name", extension.Name)
		}

		if seen[extension.Name] {
			return fmt.Errorf("pi extension name %q is used twice", extension.Name)
		}

		seen[extension.Name] = true
	}

	return nil
}

// validatePathPolicy requires absolute sandbox roots.
func validatePathPolicy(settings Settings) error {
	policy := settings.FileToolPathPolicy
	if policy == nil {
		return nil
	}

	groups := map[string][]string{
		"readable": policy.ReadableRoots,
		"writable": policy.WritableRoots,
		"denied":   policy.DeniedRoots,
	}

	for _, kind := range slices.Sorted(maps.Keys(groups)) {
		for _, root := range groups[kind] {
			if !path.IsAbs(root) {
				return fmt.Errorf("pi %s root must be an absolute sandbox path: %s", kind, root)
			}
		}
	}

	return nil
}

// validateMCPServers rejects the codemode exposures and configurations that
// do not encode to JSON.
func validateMCPServers(settings Settings) error {
	for _, name := range slices.Sorted(maps.Keys(settings.MCPServers)) {
		config := settings.MCPServers[name]

		if usesCodemodeExposure(config) {
			return fmt.Errorf(
				"pi mcp server %q uses the codemode exposure, which is not supported: the harness does not load Pi's codemode extension. Use 'direct', 'deferred' or 'hidden'",
				name,
			)
		}

		if _, err := json.Marshal(config); err != nil {
			return fmt.Errorf("encode pi mcp server %q: %w", name, err)
		}
	}

	return nil
}

// usesCodemodeExposure reports a server or tool exposure of codemode.
func usesCodemodeExposure(config map[string]any) bool {
	if exposure, ok := config["exposure"].(string); ok && slices.Contains(codemodeExposures, exposure) {
		return true
	}

	toolExposure, ok := config["toolExposure"].(map[string]any)
	if !ok {
		return false
	}

	for _, value := range toolExposure {
		if exposure, ok := value.(string); ok && slices.Contains(codemodeExposures, exposure) {
			return true
		}
	}

	return false
}

// validateProviders requires provider configurations that encode to JSON.
func validateProviders(settings Settings) error {
	for _, name := range slices.Sorted(maps.Keys(settings.Providers)) {
		if _, err := json.Marshal(settings.Providers[name]); err != nil {
			return fmt.Errorf("encode pi provider %q: %w", name, err)
		}
	}

	return nil
}

// mcpServerRegistrations returns the MCP servers in name order, with the
// default "direct" exposure applied.
func mcpServerRegistrations(servers map[string]map[string]any) []namedConfig {
	registrations := make([]namedConfig, 0, len(servers))

	for _, name := range slices.Sorted(maps.Keys(servers)) {
		config := maps.Clone(servers[name])
		if config == nil {
			config = make(map[string]any)
		}

		if _, ok := config["exposure"]; !ok {
			config["exposure"] = "direct"
		}

		registrations = append(registrations, namedConfig{
			Name:   name,
			Config: config,
		})
	}

	return registrations
}
