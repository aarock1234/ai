package googlevertex

import (
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/google"
)

// GoogleSearch returns the Google Search grounding tool.
func GoogleSearch(args google.SearchArgs) model.ProviderTool {
	return google.Search(args)
}

// EnterpriseWebSearch returns the Enterprise Web Search grounding tool.
func EnterpriseWebSearch() model.ProviderTool {
	return google.EnterpriseWebSearch()
}

// GoogleMaps returns the Google Maps grounding tool.
func GoogleMaps() model.ProviderTool {
	return google.Maps()
}

// URLContext returns the URL context tool.
func URLContext() model.ProviderTool {
	return google.URLContext()
}

// FileSearch returns the file search tool.
func FileSearch(args google.FileSearchArgs) model.ProviderTool {
	return google.FileSearch(args)
}

// CodeExecution returns the code execution tool.
func CodeExecution() model.ProviderTool {
	return google.CodeExecution()
}

// VertexRAGStore returns the Vertex RAG Store retrieval tool.
func VertexRAGStore(args google.VertexRAGStoreArgs) model.ProviderTool {
	return google.VertexRAGStore(args)
}
