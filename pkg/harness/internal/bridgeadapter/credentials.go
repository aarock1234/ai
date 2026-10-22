package bridgeadapter

import (
	"context"
	"fmt"
	"maps"
	"os"
	"strings"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/sandbox"
)

// HostEnvironment returns the host process environment as a map.
func HostEnvironment() map[string]string {
	environment := make(map[string]string)

	for _, entry := range os.Environ() {
		name, value, found := strings.Cut(entry, "=")
		if found && name != "" {
			environment[name] = value
		}
	}

	return environment
}

// FirstPresent returns the value of the first variable env defines, even
// when it is empty, as JavaScript's nullish coalescing does.
func FirstPresent(env map[string]string, names ...string) (string, bool) {
	for _, name := range names {
		if value, ok := env[name]; ok {
			return value, true
		}
	}

	return "", false
}

// BrokeringOptions configure BrokerCredentials.
type BrokeringOptions struct {
	Sandbox sandbox.Session

	// Environment holds the real credentials.
	Environment map[string]string

	CredentialEnvironmentVariables []string
	Forwarding                     harness.CredentialForwardingFunc

	// Previous is the sandbox credential environment recorded in the
	// lifecycle state, whose placeholders are reused.
	Previous map[string]string

	// Transformations builds the request transformations that inject the
	// real credentials in place of the placeholders of sandboxEnvironment.
	Transformations func(sandboxEnvironment map[string]string) ([]harness.RequestTransformation, error)
}

// BrokeredCredentials are the credentials of a sandbox that injects them
// outside its security boundary.
type BrokeredCredentials struct {
	// SandboxEnvironment is Environment with each credential replaced by its
	// placeholder.
	SandboxEnvironment map[string]string

	// CredentialEnvironment holds only the placeholders. Persist it in
	// lifecycle state.
	CredentialEnvironment map[string]string

	// TransformationCount is the number of installed transformations.
	TransformationCount int
}

// BrokerCredentials exposes placeholders instead of the credentials when the
// sandbox can add request transformations, and installs the transformations
// that inject the real credentials. It reports false when the sandbox
// cannot broker credentials.
func BrokerCredentials(ctx context.Context, options BrokeringOptions) (BrokeredCredentials, bool, error) {
	adder, ok := options.Sandbox.(harness.RequestTransformationAdder)
	if !ok {
		return BrokeredCredentials{}, false, nil
	}

	credentialEnvironment, err := harness.ResolveSandboxCredentialEnvironment(ctx, harness.SandboxCredentialEnvironmentOptions{
		Environment:                    options.Environment,
		CredentialEnvironmentVariables: options.CredentialEnvironmentVariables,
		Forwarding:                     options.Forwarding,
		Previous:                       options.Previous,
	})
	if err != nil {
		return BrokeredCredentials{}, false, err
	}

	sandboxEnvironment := maps.Clone(options.Environment)
	if sandboxEnvironment == nil {
		sandboxEnvironment = make(map[string]string)
	}

	maps.Copy(sandboxEnvironment, credentialEnvironment)

	transformations, err := options.Transformations(sandboxEnvironment)
	if err != nil {
		return BrokeredCredentials{}, false, err
	}

	if len(transformations) > 0 {
		if err := adder.AddRequestTransformations(ctx, transformations); err != nil {
			return BrokeredCredentials{}, false, fmt.Errorf("add credential request transformations: %w", err)
		}
	}

	return BrokeredCredentials{
		SandboxEnvironment:    sandboxEnvironment,
		CredentialEnvironment: credentialEnvironment,
		TransformationCount:   len(transformations),
	}, true, nil
}

// ForwardCredentials passes the credentials of environment through the
// forwarding function for a sandbox that cannot broker them, and warns that
// the real credentials reach the sandbox.
func ForwardCredentials(ctx context.Context, options BrokeringOptions) (map[string]string, error) {
	forwarded, err := harness.ApplyCredentialForwarding(ctx, harness.CredentialForwardingOptions{
		Environment:                    options.Environment,
		CredentialEnvironmentVariables: options.CredentialEnvironmentVariables,
		Forwarding:                     options.Forwarding,
	})
	if err != nil {
		return nil, err
	}

	harness.WarnCredentialBrokeringUnavailable(ctx, harness.CredentialBrokeringCheck{
		Environment:                    options.Environment,
		ForwardedEnvironment:           forwarded,
		CredentialEnvironmentVariables: options.CredentialEnvironmentVariables,
	})

	return forwarded, nil
}
