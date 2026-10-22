package acp

// bridgeConfigurationVariable carries the bridge configuration.
const bridgeConfigurationVariable = "AI_SDK_ACP_BRIDGE_CONFIGURATION"

// bridgeConfiguration is the JSON configuration the bridge reads from its
// environment.
type bridgeConfiguration struct {
	Authentication                *Authentication               `json:"authentication,omitempty"`
	ProviderAuthentication        *bridgeProviderAuthentication `json:"providerAuthentication,omitempty"`
	ProviderEnvironment           map[string]string             `json:"providerEnvironment,omitzero"`
	SessionMeta                   map[string]any                `json:"sessionMeta,omitzero"`
	ClientCapabilities            map[string]any                `json:"clientCapabilities,omitzero"`
	AskUserQuestionsRequestMethod string                        `json:"askUserQuestionsRequestMethod,omitempty"`
	HostToolMCPTransport          HostToolMCPTransport          `json:"hostToolMcpTransport,omitempty"`
	NonTextContentTypes           []NonTextContentType          `json:"nonTextContentTypes,omitzero"`
}

// bridgeEnvironment returns the environment variable carrying the bridge
// configuration.
func bridgeEnvironment(configuration bridgeConfiguration) (map[string]string, error) {
	encoded, err := marshalJSON(configuration)
	if err != nil {
		return nil, err
	}

	return map[string]string{
		bridgeConfigurationVariable: string(encoded),
	}, nil
}
