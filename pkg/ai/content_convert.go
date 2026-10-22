package ai

import (
	"context"
	"fmt"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// modelContentConversion describes the conversion of one model response
// into generated content.
type modelContentConversion struct {
	content []model.Content

	// toolCalls are the parsed tool calls of the response.
	toolCalls []ToolCall

	// toolOutputs are the outputs of locally executed tools.
	toolOutputs []ToolOutput

	approvalRequests  []ToolApprovalRequest
	approvalResponses []ToolApprovalResponse

	tools []Tool
	files *generatedFileResolver
}

// convertModelContent converts model content into content parts, replacing
// tool calls with their parsed form. Local tool outputs, approval requests
// and approval responses follow the model content; outputs of calls with an
// approval response come last.
func convertModelContent(ctx context.Context, conversion modelContentConversion) ([]ContentPart, error) {
	parts := make([]ContentPart, 0, len(conversion.content)+len(conversion.toolOutputs))

	for _, content := range conversion.content {
		part, err := convertModelContentPart(ctx, conversion, content)
		if err != nil {
			return nil, err
		}

		if part != nil {
			parts = append(parts, part)
		}
	}

	approvedCallIDs := make([]string, 0, len(conversion.approvalResponses))
	for _, response := range conversion.approvalResponses {
		approvedCallIDs = append(approvedCallIDs, response.ToolCall.ToolCallID)
	}

	var outputsWithApproval []ContentPart

	for _, output := range conversion.toolOutputs {
		if slices.Contains(approvedCallIDs, toolOutputCallID(output)) {
			outputsWithApproval = append(outputsWithApproval, output)
		} else {
			parts = append(parts, output)
		}
	}

	for _, request := range conversion.approvalRequests {
		parts = append(parts, request)
	}

	for _, response := range conversion.approvalResponses {
		parts = append(parts, response)
	}

	return append(parts, outputsWithApproval...), nil
}

func convertModelContentPart(ctx context.Context, conversion modelContentConversion, content model.Content) (ContentPart, error) {
	switch typed := content.(type) {
	case model.Text:
		return TextContent{
			Text:             typed.Text,
			Citations:        typed.Citations,
			ProviderMetadata: typed.ProviderMetadata,
		}, nil

	case model.Reasoning:
		return ReasoningContent{
			Text:             typed.Text,
			ProviderMetadata: typed.ProviderMetadata,
		}, nil

	case model.Custom:
		return CustomContent{
			Kind:             typed.Kind,
			ProviderMetadata: typed.ProviderMetadata,
		}, nil

	case model.Source:
		return SourceContent{
			Source: typed,
		}, nil

	case model.File:
		file, err := conversion.files.resolve(ctx, typed.Data, typed.MediaType)
		if err != nil {
			return nil, err
		}

		return FileContent{
			File:             file,
			ProviderMetadata: typed.ProviderMetadata,
		}, nil

	case model.ReasoningFile:
		file, err := conversion.files.resolve(ctx, typed.Data, typed.MediaType)
		if err != nil {
			return nil, err
		}

		return ReasoningFileContent{
			File:             file,
			ProviderMetadata: typed.ProviderMetadata,
		}, nil

	case model.ToolCall:
		call, found := findToolCall(conversion.toolCalls, typed.ToolCallID)
		if !found {
			return nil, fmt.Errorf("tool call %s not found", typed.ToolCallID)
		}

		return call, nil

	case model.ToolResult:
		return convertProviderToolResult(conversion, typed)

	case model.ToolApprovalRequest:
		call, found := findToolCall(conversion.toolCalls, typed.ToolCallID)
		if !found {
			return nil, &ToolCallNotFoundForApprovalError{
				ToolCallID: typed.ToolCallID,
				ApprovalID: typed.ApprovalID,
			}
		}

		return ToolApprovalRequest{
			ApprovalID: typed.ApprovalID,
			ToolCall:   call,
		}, nil

	default:
		return nil, nil
	}
}

// convertProviderToolResult converts the result of a provider-executed tool.
// Tools that support deferred results may report results whose call arrived
// in an earlier response.
func convertProviderToolResult(conversion modelContentConversion, result model.ToolResult) (ContentPart, error) {
	call, found := findToolCall(conversion.toolCalls, result.ToolCallID)
	tool, toolFound := findTool(conversion.tools, result.ToolName)

	if !found {
		if !toolFound || !tool.SupportsDeferredResults || tool.kind() != ToolKindProvider {
			return nil, fmt.Errorf("tool call %s not found", result.ToolCallID)
		}

		call = ToolCall{
			ToolCallID:   result.ToolCallID,
			ToolName:     result.ToolName,
			Dynamic:      result.Dynamic,
			ToolMetadata: tool.Metadata,
		}
	}

	return newProviderToolOutput(call, result), nil
}

// newProviderToolOutput converts the result of a provider-executed tool call
// into a ToolResult or ToolError.
func newProviderToolOutput(call ToolCall, result model.ToolResult) ToolOutput {
	if result.IsError {
		return ToolError{
			ToolCallID: result.ToolCallID,
			ToolName:   result.ToolName,
			Input:      call.Input,
			Error: &ProviderToolError{
				Result: result.Result,
			},
			ProviderExecuted: true,
			Dynamic:          call.Dynamic,
			ProviderMetadata: result.ProviderMetadata,
			ToolMetadata:     call.ToolMetadata,
		}
	}

	return ToolResult{
		ToolCallID:       result.ToolCallID,
		ToolName:         result.ToolName,
		Input:            call.Input,
		Output:           result.Result,
		ProviderExecuted: true,
		Dynamic:          call.Dynamic,
		Preliminary:      result.Preliminary,
		ProviderMetadata: result.ProviderMetadata,
		ToolMetadata:     call.ToolMetadata,
	}
}

func findToolCall(calls []ToolCall, toolCallID string) (ToolCall, bool) {
	index := slices.IndexFunc(calls, func(call ToolCall) bool {
		return call.ToolCallID == toolCallID
	})
	if index < 0 {
		return ToolCall{}, false
	}

	return calls[index], true
}

// generatedFileResolver turns generated file data into bytes, downloading
// URLs once per step.
type generatedFileResolver struct {
	download Downloader
	cache    map[string][]byte
}

func newGeneratedFileResolver() *generatedFileResolver {
	return &generatedFileResolver{
		download: NewDownloader(DownloadOptions{}),
		cache:    make(map[string][]byte),
	}
}

func (r *generatedFileResolver) resolve(ctx context.Context, data model.FileData, mediaType string) (GeneratedFile, error) {
	switch typed := data.(type) {
	case model.FileBytes:
		return GeneratedFile{
			Data:      typed.Data,
			MediaType: mediaType,
		}, nil

	case model.FileURL:
		if typed.URL == nil {
			return GeneratedFile{}, &InvalidDataContentError{
				Message: "generated file URL is missing",
			}
		}

		location := typed.URL.String()
		if cached, ok := r.cache[location]; ok {
			return GeneratedFile{
				Data:      cached,
				MediaType: mediaType,
			}, nil
		}

		downloaded, err := r.download(ctx, typed.URL)
		if err != nil {
			return GeneratedFile{}, err
		}

		r.cache[location] = downloaded.Data

		return GeneratedFile{
			Data:      downloaded.Data,
			MediaType: mediaType,
		}, nil

	default:
		return GeneratedFile{}, &InvalidDataContentError{
			Message: fmt.Sprintf("unsupported generated file data type %T", data),
		}
	}
}
