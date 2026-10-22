package bedrock

import (
	"os"
	"strings"

	"github.com/aarock1234/ai/internal/awssigv4"
)

// endpointVariable overrides the endpoint of every AWS service.
const endpointVariable = "AWS_ENDPOINT_URL"

// Endpoint is a Bedrock service and the environment variable that overrides
// its URL.
type Endpoint struct {
	Service  string
	Variable string
}

// Bedrock service endpoints.
var (
	// Runtime serves Converse, InvokeModel and their streaming variants.
	Runtime = Endpoint{
		Service:  "bedrock-runtime",
		Variable: "AWS_ENDPOINT_URL_BEDROCK_RUNTIME",
	}

	// AgentRuntime serves reranking.
	AgentRuntime = Endpoint{
		Service:  "bedrock-agent-runtime",
		Variable: "AWS_ENDPOINT_URL_BEDROCK_AGENT_RUNTIME",
	}
)

// partitionSuffix is the DNS suffix of the AWS partition whose regions start
// with prefix.
type partitionSuffix struct {
	prefix string
	suffix string
}

var partitionSuffixes = []partitionSuffix{
	{
		prefix: "cn-",
		suffix: "amazonaws.com.cn",
	},
	{
		prefix: "us-iso-",
		suffix: "c2s.ic.gov",
	},
	{
		prefix: "us-isob-",
		suffix: "sc2s.sgov.gov",
	},
	{
		prefix: "eu-isoe-",
		suffix: "cloud.adc-e.uk",
	},
	{
		prefix: "us-isof-",
		suffix: "csp.hci.ic.gov",
	},
	{
		prefix: "eusc-",
		suffix: "amazonaws.eu",
	},
}

// ResolveBaseURL returns the configured base URL, the endpoint's environment
// override, AWS_ENDPOINT_URL, or the regional endpoint, without a trailing
// slash.
func (c Config) ResolveBaseURL(endpoint Endpoint) (string, error) {
	for _, candidate := range []string{c.BaseURL, os.Getenv(endpoint.Variable), os.Getenv(endpointVariable)} {
		if candidate != "" {
			return strings.TrimSuffix(candidate, "/"), nil
		}
	}

	region, err := awssigv4.LoadHostRegion(c.Region)
	if err != nil {
		return "", err
	}

	return "https://" + endpoint.Service + "." + region + "." + dnsSuffix(region), nil
}

// LoadRegion returns the configured region or AWS_REGION.
func (c Config) LoadRegion() (string, error) {
	return awssigv4.LoadRegion(c.Region)
}

func dnsSuffix(region string) string {
	for _, partition := range partitionSuffixes {
		if strings.HasPrefix(region, partition.prefix) {
			return partition.suffix
		}
	}

	return "amazonaws.com"
}
