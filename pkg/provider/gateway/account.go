package gateway

import (
	"context"
	"fmt"
	"net/url"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// Account endpoints, relative to the base URL's origin.
const (
	creditsPath    = "/v1/credits"
	reportPath     = "/v1/report"
	generationPath = "/v1/generation"
)

// Credits is the gateway credit balance of the authenticated account, as
// decimal strings.
type Credits struct {
	// Balance is the remaining credit balance.
	Balance string

	// TotalUsed is the credit consumed so far.
	TotalUsed string
}

// SpendGroup is a dimension that spend report rows are aggregated by.
type SpendGroup string

// Spend groups.
const (
	SpendByDay            SpendGroup = "day"
	SpendByUser           SpendGroup = "user"
	SpendByModel          SpendGroup = "model"
	SpendByTag            SpendGroup = "tag"
	SpendByProvider       SpendGroup = "provider"
	SpendByCredentialType SpendGroup = "credential_type"
)

// DatePart is the time granularity of date-grouped spend reports.
type DatePart string

// Date parts.
const (
	DatePartDay  DatePart = "day"
	DatePartHour DatePart = "hour"
)

// CredentialType distinguishes your own provider credentials from the
// gateway's.
type CredentialType string

// Credential types.
const (
	CredentialBYOK   CredentialType = "byok"
	CredentialSystem CredentialType = "system"
)

// SpendReportParams select and aggregate a spend report. StartDate and
// EndDate are required; empty filters are not applied.
type SpendReportParams struct {
	StartDate      string
	EndDate        string
	GroupBy        SpendGroup
	DatePart       DatePart
	UserID         string
	Model          string
	Provider       string
	CredentialType CredentialType
	Tags           []string
}

// SpendReport is the spend of each aggregated group.
type SpendReport struct {
	Results []SpendReportRow
}

// SpendReportRow is the spend of one group. The group field matching the
// report's grouping is set; nil counts were not reported.
type SpendReportRow struct {
	Day            string         `json:"day,omitempty"`
	Hour           string         `json:"hour,omitempty"`
	User           string         `json:"user,omitempty"`
	Model          string         `json:"model,omitempty"`
	Tag            string         `json:"tag,omitempty"`
	Provider       string         `json:"provider,omitempty"`
	CredentialType CredentialType `json:"credential_type,omitempty"`

	// TotalCost is the cost in USD.
	TotalCost float64 `json:"total_cost"`

	// MarketCost is the cost at the providers' list prices in USD.
	MarketCost *float64 `json:"market_cost,omitempty"`

	InputTokens              *float64 `json:"input_tokens,omitempty"`
	OutputTokens             *float64 `json:"output_tokens,omitempty"`
	CachedInputTokens        *float64 `json:"cached_input_tokens,omitempty"`
	CacheCreationInputTokens *float64 `json:"cache_creation_input_tokens,omitempty"`
	ReasoningTokens          *float64 `json:"reasoning_tokens,omitempty"`
	RequestCount             *float64 `json:"request_count,omitempty"`
}

// GenerationInfo details one generation.
type GenerationInfo struct {
	ID string `json:"id"`

	// TotalCost and UpstreamInferenceCost are in USD.
	TotalCost             float64 `json:"total_cost"`
	UpstreamInferenceCost float64 `json:"upstream_inference_cost"`

	Usage        float64 `json:"usage"`
	CreatedAt    string  `json:"created_at"`
	Model        string  `json:"model"`
	IsBYOK       bool    `json:"is_byok"`
	ProviderName string  `json:"provider_name"`
	Streamed     bool    `json:"streamed"`
	FinishReason string  `json:"finish_reason"`

	// Latency and GenerationTime are in milliseconds.
	Latency        float64 `json:"latency"`
	GenerationTime float64 `json:"generation_time"`

	PromptTokens           float64 `json:"native_tokens_prompt"`
	CompletionTokens       float64 `json:"native_tokens_completion"`
	ReasoningTokens        float64 `json:"native_tokens_reasoning"`
	CachedTokens           float64 `json:"native_tokens_cached"`
	CacheCreationTokens    float64 `json:"native_tokens_cache_creation"`
	BillableWebSearchCalls float64 `json:"billable_web_search_calls"`
}

type creditsResponse struct {
	Balance   string `json:"balance"`
	TotalUsed string `json:"total_used"`
}

type spendReportResponse struct {
	Results []SpendReportRow `json:"results"`
}

type generationResponse struct {
	Data *GenerationInfo `json:"data"`
}

// Credits returns the credit balance. A configured team ID or slug selects
// the team.
func (p *Provider) Credits(ctx context.Context) (*Credits, error) {
	endpoint, err := p.origin()
	if err != nil {
		return nil, err
	}

	endpoint.Path = creditsPath

	if team := strings.TrimSpace(p.teamHeader()); team != "" {
		key := "slug"
		if strings.HasPrefix(team, "team_") {
			key = "teamId"
		}

		endpoint.RawQuery = url.Values{
			key: {
				team,
			},
		}.Encode()
	}

	decoded, err := getAccount[creditsResponse](ctx, p, endpoint.String())
	if err != nil {
		return nil, err
	}

	return &Credits{
		Balance:   decoded.Balance,
		TotalUsed: decoded.TotalUsed,
	}, nil
}

// teamHeader returns the team of the configured headers, which may set it
// directly, or the configured team ID or slug.
func (p *Provider) teamHeader() string {
	if team := p.headers.Get(teamHeader); team != "" {
		return team
	}

	return p.teamIDOrSlug
}

// SpendReport returns cost, token and request counts aggregated by the
// requested group.
func (p *Provider) SpendReport(ctx context.Context, params SpendReportParams) (*SpendReport, error) {
	if params.StartDate == "" || params.EndDate == "" {
		return nil, fmt.Errorf("%w: spend report start and end dates are required", model.ErrInvalidArgument)
	}

	endpoint, err := p.origin()
	if err != nil {
		return nil, err
	}

	endpoint.Path = reportPath
	endpoint.RawQuery = params.query().Encode()

	decoded, err := getAccount[spendReportResponse](ctx, p, endpoint.String())
	if err != nil {
		return nil, err
	}

	return &SpendReport{
		Results: decoded.Results,
	}, nil
}

func (s SpendReportParams) query() url.Values {
	query := url.Values{
		"start_date": {
			s.StartDate,
		},
		"end_date": {
			s.EndDate,
		},
	}

	optional := map[string]string{
		"group_by":        string(s.GroupBy),
		"date_part":       string(s.DatePart),
		"user_id":         s.UserID,
		"model":           s.Model,
		"provider":        s.Provider,
		"credential_type": string(s.CredentialType),
		"tags":            strings.Join(s.Tags, ","),
	}

	for key, value := range optional {
		if value != "" {
			query.Set(key, value)
		}
	}

	return query
}

// GenerationInfo returns the cost, usage, latency and provider details of a
// generation.
func (p *Provider) GenerationInfo(ctx context.Context, generationID string) (*GenerationInfo, error) {
	endpoint, err := p.origin()
	if err != nil {
		return nil, err
	}

	endpoint.Path = generationPath
	endpoint.RawQuery = url.Values{
		"id": {
			generationID,
		},
	}.Encode()

	decoded, err := getAccount[generationResponse](ctx, p, endpoint.String())
	if err != nil {
		return nil, err
	}

	if decoded.Data == nil {
		return nil, fmt.Errorf("%w: generation response has no data", model.ErrInvalidResponse)
	}

	return decoded.Data, nil
}

// getAccount fetches an account endpoint and decodes its JSON body.
func getAccount[Response any](ctx context.Context, p *Provider, endpoint string) (Response, error) {
	var decoded Response

	request, method, err := p.accountRequest(endpoint)
	if err != nil {
		return decoded, err
	}

	response, err := apicall.Get(ctx, request)
	if err != nil {
		return decoded, asGatewayError(err, method)
	}

	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return decoded, err
	}

	return decoded, nil
}
