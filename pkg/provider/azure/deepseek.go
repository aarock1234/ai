package azure

import (
	"net/http"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/provider/deepseek"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

// DeepSeekOptions are the options of an Azure-hosted DeepSeek call, read
// from ProviderOptions["azure"]. Azure does not support Thinking.
type DeepSeekOptions = deepseek.ChatOptions

// DeepSeek returns an Azure-hosted DeepSeek chat model for a deployment.
// Penalties and JSON schema response formats are sent; thinking settings are
// not, because Azure returns JSON in the reasoning content when thinking is
// enabled.
func (p *Provider) DeepSeek(deploymentID string) *deepseek.ChatModel {
	s := &p.settings

	return deepseek.NewChatModel(deploymentID, deepseek.ChatConfig{
		Provider: s.name + ".deepseek",
		BuildURL: func(path string) (string, error) {
			return s.deploymentURL(deploymentID, path)
		},
		Headers:                   s.deploymentHeaders,
		HTTPClient:                s.client(),
		GenerateID:                s.generateID,
		SupportsPenaltySampling:   true,
		SupportsStructuredOutputs: true,
		ThinkingUnsupported:       true,
	})
}

// deploymentURL returns the OpenAI API URL of a deployment path.
func (s *settings) deploymentURL(deploymentID string, path string) (string, error) {
	return s.openaiURL(openai.Endpoint{
		Path:    path,
		ModelID: deploymentID,
	})
}

// deploymentHeaders returns the authentication and provider headers.
func (s *settings) deploymentHeaders() (http.Header, error) {
	authentication, err := s.authHeaders(apiOpenAI)
	if err != nil {
		return nil, err
	}

	return apicall.MergeHeaders(authentication, s.headers), nil
}
