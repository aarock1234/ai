// Package policyopa authorizes tool calls with policy as code, a port of the
// Vercel AI SDK @ai-sdk/policy-opa package.
//
// Rules live in Rego and are evaluated by Open Policy Agent. OPAPolicy turns
// a policy into an ai.ToolApprovalFunc for GenerateTextOptions.ToolApproval,
// StreamTextOptions.ToolApproval or an agent:
//
//	client := policyopa.NewHTTPPolicyClient(policyopa.HTTPPolicyClientOptions{
//		URL: "http://localhost:8181",
//	})
//
//	toolApproval := policyopa.OPAPolicy(policyopa.OPAPolicyOptions{
//		Client: client,
//		Path:   "agent/call/decision",
//	})
//
// Shadow evaluates a policy without enforcing it, WrapMCPTools sends every
// tool the policy has no opinion on through a fallback decision, and
// OPACapabilityMiddleware removes the tools a policy does not allow from
// model calls. Shadow and WrapMCPTools accept any
// ai.ToolApprovalConfiguration, including per-tool ai.ToolApprovals maps.
//
// Evaluation errors fail closed: OPAPolicy denies the call and
// OPACapabilityMiddleware removes every tool. Canceled contexts are the
// exception; their error is returned so that the operation stops.
//
// Go adaptations: policies receive the Go context instead of a runtime
// context, so the default OPA input has no runtimeContext member; use
// ToInput to add request-scoped values. The in-process WebAssembly client of
// upstream lives in the nested module
// github.com/aarock1234/ai/policyopawasm, so that this module does not depend
// on a WebAssembly runtime.
package policyopa

import (
	"context"
	"encoding/json"

	"github.com/aarock1234/ai/internal/wirejson"
)

// PolicyClient evaluates a policy decision in an external engine, such as
// OPA. Adapters can switch engines without changing call sites.
type PolicyClient interface {
	// Evaluate evaluates input against the policy at path and returns the
	// JSON decision the engine produced, or nil when the policy produced no
	// decision.
	Evaluate(ctx context.Context, path string, input any) (json.RawMessage, error)
}

// policyJSON encodes messages for the OPA input in the upstream model
// message format, with file contents flattened to strings.
var policyJSON = wirejson.Encoder{
	FileData: wirejson.FlattenedFileData,
}
