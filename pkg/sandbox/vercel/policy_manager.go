package vercel

import (
	"context"
	"encoding/json"
	"maps"
	"slices"
	"strings"
	"sync"

	"github.com/aarock1234/ai/pkg/harness"
)

// accessPolicy is the network access part of a policy.
type accessPolicy struct {
	mode         harness.NetworkPolicyMode
	allowedHosts []string
	allowedCIDRs []string
	deniedCIDRs  []string
}

// forwardEntry is a forwarding rule kept verbatim from the live policy.
type forwardEntry struct {
	host       string
	match      *ruleMatch
	forwardURL string
}

// managedPolicy is the desired state the manager owns.
type managedPolicy struct {
	access          accessPolicy
	transformations []harness.RequestTransformation
	forwards        []forwardEntry
	effective       vercelPolicy
}

// policyInspection is the current policy split into its parts.
type policyInspection struct {
	access              accessPolicy
	forwards            []forwardEntry
	transformationHosts []string
	current             vercelPolicy
}

// policyManager owns the complete network policy of one session. Vercel
// stores network access, request transformations and forwarding in one
// structure, so every mutation composes a full replacement here. Mutations
// run one at a time in call order; a failed mutation leaves the state
// unchanged.
type policyManager struct {
	mu    sync.Mutex
	state *managedPolicy

	// current returns the live policy of the session.
	current func() vercelPolicy

	// apply replaces the live policy.
	apply func(ctx context.Context, policy vercelPolicy) error
}

// setNetworkPolicy replaces the network access, keeping the managed request
// transformations and the live forwarding rules.
func (m *policyManager) setNetworkPolicy(ctx context.Context, policy harness.NetworkPolicy) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	inspection := m.inspect()

	if m.state == nil && len(inspection.transformationHosts) > 0 {
		return policyConflict("Cannot set the network policy because the current Vercel Sandbox policy contains request transformations whose redacted values cannot be preserved safely. Replace them explicitly with SetRequestTransformations or rehydrate them with AddRequestTransformations first.")
	}

	access, err := toAccessPolicy(policy)
	if err != nil {
		return err
	}

	var transformations []harness.RequestTransformation
	if m.state != nil {
		transformations = m.state.transformations
	}

	return m.applyState(ctx, managedPolicy{
		access:          access,
		transformations: transformations,
		forwards:        inspection.forwards,
	}, inspection.current)
}

// setRequestTransformations replaces every request transformation.
func (m *policyManager) setRequestTransformations(ctx context.Context, transformations []harness.RequestTransformation) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	inspection := m.inspect()

	return m.applyState(ctx, managedPolicy{
		access:          inspection.access,
		transformations: slices.Clone(transformations),
		forwards:        inspection.forwards,
	}, inspection.current)
}

// addRequestTransformations adds request transformations, replacing managed
// ones with the same match and header names.
//
// Vercel redacts transformed header values when it reports a policy, so on
// the first mutation existing transformations can only be attributed by
// host; transformations for hosts this call does not cover cannot be
// preserved and are reported as a conflict.
func (m *policyManager) addRequestTransformations(ctx context.Context, transformations []harness.RequestTransformation) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	inspection := m.inspect()
	incoming := slices.Clone(transformations)

	if m.state == nil {
		incomingHosts := transformationHosts(composePolicy(inspection.access, incoming, inspection.forwards))

		for _, host := range inspection.transformationHosts {
			if !slices.Contains(incomingHosts, host) {
				return policyConflict("Cannot add request transformations because the current Vercel Sandbox policy contains request transformations that cannot be attributed to this call. Their header values are redacted, so preserving them safely is not possible.")
			}
		}
	}

	merged := incoming
	if m.state != nil {
		merged = mergeTransformations(m.state.transformations, incoming)
	}

	return m.applyState(ctx, managedPolicy{
		access:          inspection.access,
		transformations: merged,
		forwards:        inspection.forwards,
	}, inspection.current)
}

// inspect returns the managed state, or the live policy before the first
// mutation. The caller holds mu.
func (m *policyManager) inspect() policyInspection {
	if m.state != nil {
		return policyInspection{
			access:   m.state.access,
			forwards: m.state.forwards,
			current:  m.state.effective,
		}
	}

	return inspectPolicy(m.current())
}

// applyState composes and applies a desired state, skipping the update when
// the live policy already matches. The caller holds mu.
func (m *policyManager) applyState(ctx context.Context, desired managedPolicy, current vercelPolicy) error {
	desired.effective = composePolicy(desired.access, desired.transformations, desired.forwards)

	if policyKey(current) != policyKey(desired.effective) {
		if err := m.apply(ctx, desired.effective); err != nil {
			return err
		}
	}

	m.state = &desired

	return nil
}

// inspectPolicy splits a live policy into access, forwarding rules and the
// hosts of its request transformations.
func inspectPolicy(policy vercelPolicy) policyInspection {
	if policy.Mode == policyAllowAll || policy.Mode == policyDenyAll {
		return policyInspection{
			access: accessPolicy{
				mode: harness.NetworkPolicyMode(policy.Mode),
			},
			current: policy,
		}
	}

	allowedHosts := policy.AllowHosts
	if policy.AllowRules != nil {
		allowedHosts = slices.Sorted(maps.Keys(policy.AllowRules))
	}

	var forwards []forwardEntry
	var hosts []string

	for _, host := range slices.Sorted(maps.Keys(policy.AllowRules)) {
		for _, rule := range policy.AllowRules[host] {
			if hasTransformation(rule) && !slices.Contains(hosts, strings.ToLower(host)) {
				hosts = append(hosts, strings.ToLower(host))
			}

			if rule.ForwardURL != "" {
				forwards = append(forwards, forwardEntry{
					host:       host,
					match:      rule.Match,
					forwardURL: rule.ForwardURL,
				})
			}
		}
	}

	access := accessPolicy{
		mode:         harness.NetworkCustom,
		allowedHosts: minimizeHostPatterns(allowedHosts),
	}

	if policy.Subnets != nil {
		access.allowedCIDRs = policy.Subnets.Allow
		access.deniedCIDRs = policy.Subnets.Deny
	}

	return policyInspection{
		access:              access,
		forwards:            forwards,
		transformationHosts: hosts,
		current:             policy,
	}
}

// toAccessPolicy converts a harness network policy.
func toAccessPolicy(policy harness.NetworkPolicy) (accessPolicy, error) {
	switch policy.Mode {
	case harness.NetworkAllowAll, harness.NetworkDenyAll:
		return accessPolicy{
			mode: policy.Mode,
		}, nil
	case harness.NetworkCustom:
		if len(policy.AllowedHosts) == 0 && len(policy.AllowedCIDRs) == 0 && len(policy.DeniedCIDRs) == 0 {
			return accessPolicy{}, policyConflict("Custom network policy requires at least one of AllowedHosts, AllowedCIDRs, or DeniedCIDRs to be non-empty.")
		}

		return accessPolicy{
			mode:         harness.NetworkCustom,
			allowedHosts: slices.Clone(policy.AllowedHosts),
			allowedCIDRs: slices.Clone(policy.AllowedCIDRs),
			deniedCIDRs:  slices.Clone(policy.DeniedCIDRs),
		}, nil
	default:
		return accessPolicy{}, policyConflict("Unknown network policy mode '" + string(policy.Mode) + "'.")
	}
}

// composePolicy composes the effective policy. Access is authoritative:
// rules whose host does not intersect an allowed host are left out, but stay
// in the desired state so that a later access change can activate them.
// CIDR access never activates host rules.
func composePolicy(access accessPolicy, transformations []harness.RequestTransformation, forwards []forwardEntry) vercelPolicy {
	if access.mode == harness.NetworkDenyAll {
		return vercelPolicy{
			Mode: policyDenyAll,
		}
	}

	allowedHosts := access.allowedHosts
	if access.mode == harness.NetworkAllowAll {
		allowedHosts = []string{
			"*",
		}
	}

	rules := make(map[string][]policyRule, len(allowedHosts))
	for _, host := range allowedHosts {
		rules[host] = []policyRule{}
	}

	hasActiveRules := false

	for _, transformation := range transformations {
		for _, host := range activeHostPatterns(allowedHosts, transformation.Match.Host) {
			rules[host] = append(rules[host], transformationRule(transformation))
			hasActiveRules = true
		}
	}

	for _, forward := range forwards {
		for _, host := range activeHostPatterns(allowedHosts, forward.host) {
			rules[host] = append(rules[host], policyRule{
				Match:      forward.match,
				ForwardURL: forward.forwardURL,
			})
			hasActiveRules = true
		}
	}

	if !hasActiveRules {
		return accessOnlyPolicy(access)
	}

	policy := vercelPolicy{
		AllowRules: rules,
	}

	if access.mode == harness.NetworkCustom {
		policy.Subnets = subnetsOf(access)
	}

	return policy
}

// accessOnlyPolicy encodes access without rules.
func accessOnlyPolicy(access accessPolicy) vercelPolicy {
	if access.mode != harness.NetworkCustom {
		return vercelPolicy{
			Mode: string(access.mode),
		}
	}

	policy := vercelPolicy{
		Subnets: subnetsOf(access),
	}

	if len(access.allowedHosts) > 0 {
		policy.AllowHosts = slices.Clone(access.allowedHosts)
	}

	return policy
}

// subnetsOf returns the CIDRs of a custom access policy, or nil.
func subnetsOf(access accessPolicy) *policySubnets {
	if len(access.allowedCIDRs) == 0 && len(access.deniedCIDRs) == 0 {
		return nil
	}

	return &policySubnets{
		Allow: access.allowedCIDRs,
		Deny:  access.deniedCIDRs,
	}
}

// transformationRule converts a request transformation into a Vercel rule.
func transformationRule(transformation harness.RequestTransformation) policyRule {
	match := transformation.Match
	rule := policyRule{
		Transform: []ruleTransform{
			{
				Headers: maps.Clone(transformation.Transform.Headers),
			},
		},
	}

	hasMatch := match.Path != nil || match.Method != nil || match.QueryString != nil || match.Headers != nil
	if hasMatch {
		rule.Match = &ruleMatch{
			Path:        match.Path,
			Method:      match.Method,
			QueryString: match.QueryString,
			Headers:     match.Headers,
		}
	}

	return rule
}

// activeHostPatterns returns the intersections of the allowed hosts with a
// rule's host.
func activeHostPatterns(allowedHosts []string, ruleHost string) []string {
	var intersections []string

	for _, allowedHost := range allowedHosts {
		switch {
		case isHostPatternSubset(allowedHost, ruleHost):
			intersections = append(intersections, allowedHost)
		case isHostPatternSubset(ruleHost, allowedHost):
			intersections = append(intersections, ruleHost)
		}
	}

	return minimizeHostPatterns(intersections)
}

// isHostPatternSubset reports whether every host candidate matches is also
// matched by container. "*" matches everything and "*.example.com" every
// subdomain.
func isHostPatternSubset(candidate string, container string) bool {
	candidate = strings.ToLower(candidate)
	container = strings.ToLower(container)

	if candidate == container || container == "*" {
		return true
	}

	if candidate == "*" || !strings.HasPrefix(container, "*") {
		return false
	}

	suffix := container[1:]
	if strings.HasPrefix(candidate, "*") {
		return strings.HasSuffix(candidate[1:], suffix)
	}

	return strings.HasSuffix(candidate, suffix)
}

// minimizeHostPatterns removes duplicates and patterns another pattern
// covers.
func minimizeHostPatterns(patterns []string) []string {
	unique := slices.Compact(slices.Clone(patterns))

	var minimized []string
	for _, pattern := range unique {
		covered := slices.ContainsFunc(unique, func(other string) bool {
			return other != pattern && isHostPatternSubset(pattern, other)
		})

		if !covered && !slices.Contains(minimized, pattern) {
			minimized = append(minimized, pattern)
		}
	}

	return minimized
}

// hasTransformation reports a rule that transforms headers.
func hasTransformation(rule policyRule) bool {
	for _, transform := range rule.Transform {
		if len(transform.Headers) > 0 {
			return true
		}
	}

	return false
}

// transformationHosts returns the lowercase hosts with transformations.
func transformationHosts(policy vercelPolicy) []string {
	var hosts []string

	for host, rules := range policy.AllowRules {
		for _, rule := range rules {
			if hasTransformation(rule) && !slices.Contains(hosts, strings.ToLower(host)) {
				hosts = append(hosts, strings.ToLower(host))
			}
		}
	}

	return hosts
}

// mergeTransformations adds incoming transformations, replacing existing
// ones with the same identity in place.
func mergeTransformations(existing []harness.RequestTransformation, incoming []harness.RequestTransformation) []harness.RequestTransformation {
	merged := slices.Clone(existing)

	for _, transformation := range incoming {
		identity := transformationIdentity(transformation)

		index := slices.IndexFunc(merged, func(candidate harness.RequestTransformation) bool {
			return transformationIdentity(candidate) == identity
		})

		if index < 0 {
			merged = append(merged, transformation)
		} else {
			merged[index] = transformation
		}
	}

	return merged
}

// transformationIdentity identifies a transformation by its match and the
// names of the headers it transforms.
func transformationIdentity(transformation harness.RequestTransformation) string {
	names := make([]string, 0, len(transformation.Transform.Headers))
	for name := range transformation.Transform.Headers {
		names = append(names, strings.ToLower(name))
	}

	slices.Sort(names)

	encoded, err := json.Marshal(struct {
		Match       harness.RequestMatch `json:"match"`
		HeaderNames []string             `json:"transformedHeaderNames"`
	}{
		Match:       transformation.Match,
		HeaderNames: names,
	})
	if err != nil {
		return transformation.Match.Host
	}

	return string(encoded)
}

// policyConflict reports a policy the manager cannot apply safely.
func policyConflict(message string) error {
	return &harness.CapabilityUnsupportedError{
		Message:   message,
		HarnessID: providerID,
	}
}
