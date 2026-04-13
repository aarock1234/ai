package cohere_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/cohere"
)

func TestRerankModelPassesProviderOptions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/rerank" {
			t.Fatalf("path = %q, want /rerank", r.URL.Path)
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("ReadAll: %v", err)
		}

		var req map[string]any
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("Unmarshal: %v", err)
		}

		if req["priority"] != float64(5) {
			t.Fatalf("priority = %v, want 5", req["priority"])
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"results":[{"index":1,"relevance_score":0.9}]}`))
	}))
	defer server.Close()

	m := cohere.Rerank("rerank-v3.5",
		cohere.WithAPIKey("test-key"),
		cohere.WithBaseURL(server.URL),
	)

	resp, err := m.DoRerank(context.Background(), &model.RerankRequest{
		Query:     "hello",
		Documents: []string{"a", "b"},
		ProviderOptions: model.ProviderOptions{
			"cohere": {
				"priority": json.RawMessage(`5`),
			},
		},
	})
	if err != nil {
		t.Fatalf("DoRerank: %v", err)
	}

	if len(resp.Results) != 1 || resp.Results[0].Document != "b" {
		t.Fatalf("results = %#v, want reranked second document", resp.Results)
	}
}
