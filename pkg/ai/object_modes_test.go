package ai_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// textModel returns a model that answers every generate call with text.
func textModel(text string) *mockModel {
	return &mockModel{
		supportsResponseFormat: true,
		generateFn: func(_ context.Context, _ *model.Request) (*model.Response, error) {
			return &model.Response{
				Content:      []model.Content{model.TextContent{Text: text}},
				FinishReason: model.FinishReasonStop,
			}, nil
		},
	}
}

func TestGenerateObjectArray(t *testing.T) {
	type item struct {
		Name string `json:"name"`
	}

	mock := textModel(`{"elements":[{"name":"a"},{"name":"b"}]}`)

	result, err := ai.GenerateObjectArray[item](context.Background(), mock, ai.WithPrompt("list"))
	if err != nil {
		t.Fatalf("GenerateObjectArray: %v", err)
	}

	if len(result.Object) != 2 || result.Object[0].Name != "a" || result.Object[1].Name != "b" {
		t.Errorf("object = %+v, want [a b]", result.Object)
	}
}

func TestGenerateObjectArraySchemaEnvelope(t *testing.T) {
	type item struct {
		Name string `json:"name"`
	}

	var captured *model.Request
	mock := &mockModel{
		supportsResponseFormat: true,
		generateFn: func(_ context.Context, req *model.Request) (*model.Response, error) {
			captured = req

			return &model.Response{
				Content:      []model.Content{model.TextContent{Text: `{"elements":[]}`}},
				FinishReason: model.FinishReasonStop,
			}, nil
		},
	}

	if _, err := ai.GenerateObjectArray[item](context.Background(), mock, ai.WithPrompt("list")); err != nil {
		t.Fatalf("GenerateObjectArray: %v", err)
	}

	if captured == nil || captured.ResponseFormat == nil || captured.ResponseFormat.JSONSchema == nil {
		t.Fatal("response format not set")
	}
	schemaJSON := string(captured.ResponseFormat.JSONSchema.Schema)
	if !strings.Contains(schemaJSON, `"elements"`) || !strings.Contains(schemaJSON, `"array"`) {
		t.Errorf("schema = %s, want elements array envelope", schemaJSON)
	}
}

func TestGenerateEnum(t *testing.T) {
	mock := textModel(`{"result":"positive"}`)

	result, err := ai.GenerateEnum(context.Background(), mock,
		[]string{"positive", "negative", "neutral"},
		ai.WithPrompt("classify"),
	)
	if err != nil {
		t.Fatalf("GenerateEnum: %v", err)
	}

	if result.Object != "positive" {
		t.Errorf("object = %q, want positive", result.Object)
	}
}

func TestGenerateEnumRejectsUnknownValue(t *testing.T) {
	mock := textModel(`{"result":"sideways"}`)

	_, err := ai.GenerateEnum(context.Background(), mock,
		[]string{"up", "down"},
		ai.WithPrompt("classify"),
	)
	if !errors.Is(err, ai.ErrSchemaValidation) {
		t.Fatalf("err = %v, want ErrSchemaValidation", err)
	}
}

func TestGenerateObjectRepairText(t *testing.T) {
	type recipe struct {
		Name string `json:"name"`
	}

	// Model emits fenced JSON that fails to parse directly.
	mock := textModel("```json\n{\"name\":\"pasta\"}\n```")

	repairCalls := 0
	result, err := ai.GenerateObject[recipe](context.Background(), mock,
		ai.WithPrompt("recipe"),
		ai.WithRepairText(func(_ context.Context, raw string, _ error) (string, error) {
			repairCalls++
			cleaned := strings.TrimPrefix(strings.TrimSpace(raw), "```json")
			cleaned = strings.TrimSuffix(strings.TrimSpace(cleaned), "```")

			return strings.TrimSpace(cleaned), nil
		}),
	)
	if err != nil {
		t.Fatalf("GenerateObject: %v", err)
	}

	if repairCalls != 1 {
		t.Errorf("repair calls = %d, want 1", repairCalls)
	}
	if result.Object.Name != "pasta" {
		t.Errorf("name = %q, want pasta", result.Object.Name)
	}
}

func TestGenerateObjectParseFailureIsSchemaValidation(t *testing.T) {
	type recipe struct {
		Name string `json:"name"`
	}

	mock := textModel("not json at all")

	_, err := ai.GenerateObject[recipe](context.Background(), mock, ai.WithPrompt("recipe"))
	if !errors.Is(err, ai.ErrSchemaValidation) {
		t.Fatalf("err = %v, want ErrSchemaValidation", err)
	}
}

// TestRecursiveTypeSchema verifies self-referential types no longer
// overflow the stack during schema generation.
func TestRecursiveTypeSchema(t *testing.T) {
	type node struct {
		Value    string  `json:"value"`
		Children []*node `json:"children,omitempty"`
	}

	schema := ai.SchemaFor[node]()
	if len(schema) == 0 {
		t.Fatal("empty schema for recursive type")
	}
	if !strings.Contains(string(schema), `"children"`) {
		t.Errorf("schema = %s, want children property", schema)
	}
}
