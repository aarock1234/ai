package vercel

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/harness"
)

// Vercel network policy modes.
const (
	policyAllowAll = "allow-all"
	policyDenyAll  = "deny-all"
)

// redactedHeaderValue replaces transformed header values the API redacts.
const redactedHeaderValue = "<redacted>"

// vercelPolicy is a Vercel Sandbox network policy in the request shape of
// the API: a mode, or an allow list or map of per-host rules with subnets.
// Vercel combines network access, request transformations and forwarding in
// this one structure.
type vercelPolicy struct {
	// Mode is policyAllowAll or policyDenyAll; empty selects the object form.
	Mode string

	// AllowHosts is the allow list form; AllowRules the per-host rules form.
	// At most one is set.
	AllowHosts []string
	AllowRules map[string][]policyRule

	Subnets *policySubnets
}

// policyRule is one per-host rule: a request transformation or a forward.
type policyRule struct {
	Match      *ruleMatch      `json:"match,omitempty"`
	Transform  []ruleTransform `json:"transform,omitempty"`
	ForwardURL string          `json:"forwardURL,omitempty"`
}

// ruleMatch selects the requests of a rule.
type ruleMatch struct {
	Path        *harness.Matcher          `json:"path,omitempty"`
	Method      []string                  `json:"method,omitempty"`
	QueryString []harness.KeyValueMatcher `json:"queryString,omitempty"`
	Headers     []harness.KeyValueMatcher `json:"headers,omitempty"`
}

// ruleTransform overwrites request headers.
type ruleTransform struct {
	Headers map[string]string `json:"headers,omitempty"`
}

// policySubnets are the allowed and denied CIDRs.
type policySubnets struct {
	Allow []string `json:"allow,omitempty"`
	Deny  []string `json:"deny,omitempty"`
}

// MarshalJSON encodes the request shape.
func (p vercelPolicy) MarshalJSON() ([]byte, error) {
	if p.Mode != "" {
		return json.Marshal(map[string]string{
			"mode": p.Mode,
		})
	}

	encoded := map[string]any{}

	switch {
	case p.AllowRules != nil:
		encoded["allow"] = p.AllowRules
	case p.AllowHosts != nil:
		encoded["allow"] = p.AllowHosts
	}

	if p.Subnets != nil {
		encoded["subnets"] = p.Subnets
	}

	return json.Marshal(encoded)
}

// apiNetworkPolicy is the response shape of a network policy, where custom
// policies list domains, CIDRs, injection rules with redacted header values,
// and forward rules.
type apiNetworkPolicy struct {
	Mode           string          `json:"mode"`
	AllowedDomains []string        `json:"allowedDomains"`
	AllowedCIDRs   []string        `json:"allowedCIDRs"`
	DeniedCIDRs    []string        `json:"deniedCIDRs"`
	InjectionRules []injectionRule `json:"injectionRules"`
	ForwardRules   []forwardRule   `json:"forwardRules"`
}

// injectionRule is a request transformation as the API reports it.
type injectionRule struct {
	Domain      string     `json:"domain"`
	HeaderNames []string   `json:"headerNames"`
	Match       *ruleMatch `json:"match"`
}

// forwardRule is a forwarding rule as the API reports it.
type forwardRule struct {
	Domain     string     `json:"domain"`
	ForwardURL string     `json:"forwardURL"`
	Match      *ruleMatch `json:"match"`
}

// fromAPINetworkPolicy converts the response shape into the request shape,
// as the upstream SDK does. Transformed header values come back redacted.
func fromAPINetworkPolicy(api *apiNetworkPolicy) vercelPolicy {
	if api == nil {
		return vercelPolicy{
			Mode: policyAllowAll,
		}
	}

	if api.Mode == policyAllowAll || api.Mode == policyDenyAll {
		return vercelPolicy{
			Mode: api.Mode,
		}
	}

	policy := vercelPolicy{}
	if len(api.AllowedCIDRs) > 0 || len(api.DeniedCIDRs) > 0 {
		policy.Subnets = &policySubnets{
			Allow: api.AllowedCIDRs,
			Deny:  api.DeniedCIDRs,
		}
	}

	if len(api.InjectionRules) == 0 && len(api.ForwardRules) == 0 {
		policy.AllowHosts = api.AllowedDomains

		return policy
	}

	policy.AllowRules = make(map[string][]policyRule)

	for _, domain := range api.AllowedDomains {
		policy.AllowRules[domain] = []policyRule{}
	}

	for _, rule := range api.InjectionRules {
		headers := make(map[string]string, len(rule.HeaderNames))
		for _, name := range rule.HeaderNames {
			headers[name] = redactedHeaderValue
		}

		policy.AllowRules[rule.Domain] = append(policy.AllowRules[rule.Domain], policyRule{
			Match: rule.Match,
			Transform: []ruleTransform{
				{
					Headers: headers,
				},
			},
		})
	}

	for _, rule := range api.ForwardRules {
		policy.AllowRules[rule.Domain] = append(policy.AllowRules[rule.Domain], policyRule{
			Match:      rule.Match,
			ForwardURL: rule.ForwardURL,
		})
	}

	return policy
}

// policyKey serializes a policy for equality checks; map keys are sorted by
// the JSON encoder.
func policyKey(policy vercelPolicy) string {
	encoded, err := json.Marshal(policy)
	if err != nil {
		return fmt.Sprintf("%#v", policy)
	}

	return string(encoded)
}
