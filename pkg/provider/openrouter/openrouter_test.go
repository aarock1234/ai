package openrouter_test

import (
	"testing"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openrouter"
)

func TestChatDoesNotAssumeNativeStructuredOutputSupport(t *testing.T) {
	m := openrouter.Chat("openai/gpt-5.4")

	support, ok := m.(model.ResponseFormatSupport)
	if !ok {
		t.Fatal("chat model does not implement ResponseFormatSupport")
	}

	if support.SupportsResponseFormat(&model.ResponseFormat{Type: "json_schema"}) {
		t.Fatal("SupportsResponseFormat returned true, want false")
	}
}
