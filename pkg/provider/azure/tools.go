package azure

import (
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

// CodeInterpreter returns the OpenAI code interpreter tool for Responses
// models.
func CodeInterpreter(args openai.CodeInterpreterArgs) model.ProviderTool {
	return openai.CodeInterpreter(args)
}

// FileSearch returns the OpenAI file search tool for Responses models.
func FileSearch(args openai.FileSearchArgs) model.ProviderTool {
	return openai.FileSearch(args)
}

// ImageGeneration returns the OpenAI image generation tool for Responses
// models.
func ImageGeneration(args openai.ImageGenerationArgs) model.ProviderTool {
	return openai.ImageGeneration(args)
}

// WebSearch returns the OpenAI web search tool for Responses models.
func WebSearch(args openai.WebSearchArgs) model.ProviderTool {
	return openai.WebSearch(args)
}

// WebSearchPreview returns the OpenAI web search preview tool for Responses
// models.
//
// Deprecated: Use WebSearch.
func WebSearchPreview(args openai.WebSearchPreviewArgs) model.ProviderTool {
	return openai.WebSearchPreview(args)
}
