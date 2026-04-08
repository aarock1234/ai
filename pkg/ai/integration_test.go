//go:build integration

package integration_test

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openai"
	"github.com/aarock1234/ai/pkg/provider/openrouter"
)

func requireEnv(t *testing.T, key string) string {
	t.Helper()

	val := os.Getenv(key)
	if val == "" {
		t.Skipf("skipping: %s not set", key)
	}

	return val
}

func TestOpenAIGenerateText(t *testing.T) {
	requireEnv(t, "OPENAI_API_KEY")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	m := openai.Chat("gpt-4o-mini")

	result, err := ai.GenerateText(ctx, m,
		ai.WithPrompt("Say 'hello' and nothing else."),
		ai.WithMaxTokens(10),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	if result.Text == "" {
		t.Error("expected non-empty response")
	}

	if result.FinishReason != model.FinishReasonStop {
		t.Errorf("FinishReason = %v, want stop", result.FinishReason)
	}

	t.Logf("Response: %q (tokens: %d in, %d out)", result.Text, result.Usage.InputTokens, result.Usage.OutputTokens)
}

func TestOpenAIStreamText(t *testing.T) {
	requireEnv(t, "OPENAI_API_KEY")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	m := openai.Chat("gpt-4o-mini")

	stream, err := ai.StreamText(ctx, m,
		ai.WithPrompt("Count from 1 to 5."),
		ai.WithMaxTokens(50),
	)
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	var chunks int
	var text string
	for delta := range stream.Text() {
		chunks++
		text += delta
	}

	if chunks == 0 {
		t.Error("expected at least one chunk")
	}

	if text == "" {
		t.Error("expected non-empty streamed text")
	}

	t.Logf("Streamed %d chunks: %q", chunks, text)
}

func TestOpenAIGenerateObject(t *testing.T) {
	requireEnv(t, "OPENAI_API_KEY")

	type Color struct {
		Name string `json:"name" jsonschema:"description=The color name"`
		Hex  string `json:"hex" jsonschema:"description=The hex code"`
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	m := openai.Chat("gpt-4o-mini")

	result, err := ai.GenerateObject[Color](ctx, m,
		ai.WithPrompt("Give me the color red."),
	)
	if err != nil {
		t.Fatalf("GenerateObject: %v", err)
	}

	if result.Object.Name == "" {
		t.Error("expected non-empty color name")
	}

	t.Logf("Color: %s (%s)", result.Object.Name, result.Object.Hex)
}

func TestOpenAIEmbed(t *testing.T) {
	requireEnv(t, "OPENAI_API_KEY")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	m := openai.Embedding("text-embedding-3-small")

	result, err := ai.Embed(ctx, m, "hello world")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if len(result.Embedding) == 0 {
		t.Error("expected non-empty embedding")
	}

	t.Logf("Embedding dimensions: %d", len(result.Embedding))
}

func TestOpenRouterGenerateText(t *testing.T) {
	requireEnv(t, "OPENROUTER_API_KEY")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	m := openrouter.Chat("openai/gpt-4o-mini")

	result, err := ai.GenerateText(ctx, m,
		ai.WithPrompt("Say 'hello' and nothing else."),
		ai.WithMaxTokens(10),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	if result.Text == "" {
		t.Error("expected non-empty response")
	}

	t.Logf("Response: %q", result.Text)
}
