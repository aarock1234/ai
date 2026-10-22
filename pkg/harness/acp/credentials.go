package acp

import (
	"context"
	"fmt"
	"maps"

	"github.com/aarock1234/ai/pkg/harness"
)

// credentialPlan is how a session's credentials reach the sandbox.
type credentialPlan struct {
	// implementationEnv holds the real host values of the implementation's
	// environment.
	implementationEnv map[string]string

	// sandboxImplementationEnv is the implementation's environment as the
	// sandbox sees it, before credential forwarding.
	sandboxImplementationEnv map[string]string

	// providerAuthentication is the provider routing; sandboxProviderEnv is
	// the part of its environment the bridge process receives.
	providerAuthentication resolvedProviderAuthentication
	sandboxProviderEnv     map[string]string

	// providerEnvInjected reports that the gateway launch environment was
	// resolved on the host and merged into sandboxImplementationEnv.
	providerEnvInjected bool

	// forwardingNames name the variables holding credentials.
	forwardingNames []string

	// sandboxCredentialEnv holds the placeholders of brokered credentials.
	// It is nil when the sandbox cannot broker credentials.
	sandboxCredentialEnv map[string]string
}

// resolveCredentials resolves the authentication environment, the session
// identities and the credential plan, and checks lifecycle state against
// the identities.
func (s *sessionStartup) resolveCredentials(ctx context.Context) error {
	settings := s.harness.settings
	env := hostEnvironment()

	authEnv, err := s.harness.authenticationEnvironment(ctx, env)
	if err != nil {
		return err
	}

	compatibility := resolveProviderAuthenticationCompatibility(settings.Auth, settings.ProviderAuthentication, authEnv)

	identity, err := implementationIdentity(identityInput{
		harnessID:              settings.HarnessID,
		implementation:         s.harness.implementation,
		clientApp:              s.harness.clientApp,
		clientCapabilities:     settings.ClientCapabilities,
		modelMapping:           settings.ModelMapping,
		providerAuthentication: compatibility,
		permissionModeMapping:  settings.PermissionModeMapping,
		nonTextContentTypes:    settings.NonTextContentTypes,
	})
	if err != nil {
		return err
	}

	profile, err := newAuthenticationProfile(settings.Authentication, compatibility)
	if err != nil {
		return err
	}

	s.implementationIdentity = identity
	s.configuration = s.harness.sessionConfiguration(profile)

	if err := s.validateLifecycleCompatibility(profile); err != nil {
		return err
	}

	providerAuthentication, err := resolveProviderAuthentication(providerAuthenticationInput{
		auth:                   settings.Auth,
		providerAuthentication: settings.ProviderAuthentication,
		clientApp:              s.harness.clientApp,
		env:                    authEnv,
		compatibility:          compatibility,
	})
	if err != nil {
		return err
	}

	implementationEnv := s.harness.implementation.environment(environmentInput{
		env:           mergeEnvironments(env, authEnv),
		credentialEnv: authEnv,
	})

	return s.planCredentials(ctx, implementationEnv, providerAuthentication)
}

// validateLifecycleCompatibility rejects lifecycle state of another harness,
// implementation, authentication profile or sandbox.
func (s *sessionStartup) validateLifecycleCompatibility(profile authenticationProfile) error {
	if s.lifecycle == nil {
		return nil
	}

	return s.lifecycle.validateCompatibility(compatibilityCheck{
		harnessID:              s.harness.settings.HarnessID,
		stateHarnessID:         s.stateHarnessID,
		implementationIdentity: s.implementationIdentity,
		authenticationProfile:  profile,
		sandboxID:              s.sandboxID,
	})
}

// planCredentials decides how credentials reach the sandbox: brokered
// through request transformations when the sandbox supports them, and
// forwarded otherwise.
func (s *sessionStartup) planCredentials(ctx context.Context, implementationEnv map[string]string, providerAuthentication resolvedProviderAuthentication) error {
	settings := s.harness.settings

	credentialNames := uniqueStrings(settings.CredentialEnv, []string{
		sourceAIGatewayAPIKey,
		sourceVercelOIDCToken,
	})

	plan := credentialPlan{
		implementationEnv:        implementationEnv,
		sandboxImplementationEnv: implementationEnv,
		providerAuthentication:   providerAuthentication,
		sandboxProviderEnv:       providerAuthentication.env,
		forwardingNames:          uniqueStrings(credentialNames, providerCredentialNames(providerAuthentication)),
	}

	adder, canBroker := s.options.SandboxSession.(harness.RequestTransformationAdder)
	if settings.CredentialBrokering != nil && canBroker {
		if err := s.brokerCredentials(ctx, &plan, adder); err != nil {
			return err
		}
	}

	if settings.CredentialForwarding != nil && !plan.providerEnvInjected && providerAuthentication.isGateway() {
		providerEnv, err := gatewayLaunchEnvironment(providerAuthentication, s.harness.clientApp)
		if err != nil {
			return err
		}

		plan.injectProviderEnvironment(providerEnv)
	}

	s.credentials = plan

	return nil
}

// brokerCredentials replaces the credentials with placeholders and installs
// the request transformations that inject the real values.
func (s *sessionStartup) brokerCredentials(ctx context.Context, plan *credentialPlan, adder harness.RequestTransformationAdder) error {
	settings := s.harness.settings

	providerEnv, err := gatewayLaunchEnvironment(plan.providerAuthentication, s.harness.clientApp)
	if err != nil {
		return err
	}

	brokeringEnv := mergeEnvironments(plan.implementationEnv, providerEnv)

	var previous map[string]string
	if s.lifecycle != nil {
		previous = s.lifecycle.SandboxCredentialEnvironment
	}

	sandboxCredentialEnv, err := harness.ResolveSandboxCredentialEnvironment(ctx, harness.SandboxCredentialEnvironmentOptions{
		Environment:                    brokeringEnv,
		CredentialEnvironmentVariables: plan.forwardingNames,
		Forwarding:                     settings.CredentialForwarding,
		Previous:                       previous,
	})
	if err != nil {
		return err
	}

	plan.sandboxCredentialEnv = sandboxCredentialEnv
	plan.sandboxImplementationEnv = mergeEnvironments(brokeringEnv, sandboxCredentialEnv)

	transformations, err := settings.CredentialBrokering(CredentialBrokeringEvent{
		Env:        brokeringEnv,
		SandboxEnv: plan.sandboxImplementationEnv,
		Headers:    headerValues(s.options.Headers),
	})
	if err != nil {
		return fmt.Errorf("acp credential brokering: %w", err)
	}

	if len(transformations) > 0 {
		if err := adder.AddRequestTransformations(ctx, transformations); err != nil {
			return fmt.Errorf("add acp credential request transformations: %w", err)
		}
	}

	if plan.providerAuthentication.isGateway() {
		plan.injectProviderEnvironment(map[string]string{})
	}

	return nil
}

// injectProviderEnvironment merges the resolved gateway launch environment
// into the implementation's environment, so that the bridge receives no
// gateway credential of its own.
func (p *credentialPlan) injectProviderEnvironment(providerEnv map[string]string) {
	p.providerEnvInjected = true
	p.sandboxImplementationEnv = mergeEnvironments(p.sandboxImplementationEnv, providerEnv)

	sandboxProviderEnv := maps.Clone(p.providerAuthentication.env)
	delete(sandboxProviderEnv, gatewayAPIKeyVariable)
	delete(sandboxProviderEnv, gatewayBaseURLVariable)

	p.sandboxProviderEnv = sandboxProviderEnv
}

// forwardedEnvironment returns the implementation's environment after
// credential forwarding, warning when real credentials must be forwarded
// because the sandbox cannot broker them.
func (s *sessionStartup) forwardedEnvironment(ctx context.Context) (map[string]string, error) {
	plan := s.credentials
	settings := s.harness.settings

	if plan.sandboxCredentialEnv != nil {
		return plan.sandboxImplementationEnv, nil
	}

	forwarded, err := harness.ApplyCredentialForwarding(ctx, harness.CredentialForwardingOptions{
		Environment:                    plan.sandboxImplementationEnv,
		CredentialEnvironmentVariables: plan.forwardingNames,
		Forwarding:                     settings.CredentialForwarding,
	})
	if err != nil {
		return nil, err
	}

	if settings.CredentialBrokering != nil {
		harness.WarnCredentialBrokeringUnavailable(ctx, harness.CredentialBrokeringCheck{
			Environment:          mergeEnvironments(plan.sandboxImplementationEnv, plan.providerAuthentication.env),
			ForwardedEnvironment: mergeEnvironments(forwarded, plan.sandboxProviderEnv),
			CredentialEnvironmentVariables: uniqueStrings(plan.forwardingNames, []string{
				gatewayAPIKeyVariable,
			}),
		})
	}

	return forwarded, nil
}

// providerCredentialNames names the gateway launch variables that carry the
// gateway credential.
func providerCredentialNames(providerAuthentication resolvedProviderAuthentication) []string {
	if !providerAuthentication.isGateway() {
		return nil
	}

	var names []string

	for name, value := range providerAuthentication.provider.Env {
		if containsProviderCredential(value) {
			names = append(names, name)
		}
	}

	return sortedUnique(names)
}

// authenticationEnvironment resolves the environment credentials are
// discovered from.
func (h *Harness) authenticationEnvironment(ctx context.Context, env map[string]string) (map[string]string, error) {
	resolve := h.settings.ResolveAuthenticationEnvironment
	if resolve == nil {
		return authenticationEnvironment(h.settings.Auth, env), nil
	}

	resolved, err := resolve(ctx, AuthenticationEnvironmentEvent{
		Auth: h.settings.Auth,
		Env:  env,
	})
	if err != nil {
		return nil, fmt.Errorf("resolve acp authentication environment: %w", err)
	}

	return resolved, nil
}

// sessionConfiguration returns the non-secret session configuration.
func (h *Harness) sessionConfiguration(profile authenticationProfile) sessionConfiguration {
	return sessionConfiguration{
		authenticationProfile: profile,
		sessionMeta:           h.settings.Session.Meta,
		instructionMapping:    h.settings.InstructionMapping,
		outputSchemaMapping:   h.settings.OutputSchemaMapping,
		modelMapping:          h.settings.ModelMapping,
		builtinTools:          builtinToolMappings(h.builtinTools),
		permissionModeMapping: h.settings.PermissionModeMapping,
		mcpServers:            h.settings.MCPServers,
	}
}
