package schema_test

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/aarock1234/ai/pkg/schema"
)

type flatStruct struct {
	Name  string `json:"name"`
	Age   int    `json:"age"`
	Email string `json:"email"`
}

func TestGenerateFlatStruct(t *testing.T) {
	raw := schema.Generate[flatStruct]()

	var s map[string]any
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if s["type"] != "object" {
		t.Errorf("type = %v, want object", s["type"])
	}

	props, ok := s["properties"].(map[string]any)
	if !ok {
		t.Fatal("properties not a map")
	}

	if len(props) != 3 {
		t.Errorf("properties count = %d, want 3", len(props))
	}

	nameSchema := props["name"].(map[string]any)
	if nameSchema["type"] != "string" {
		t.Errorf("name type = %v, want string", nameSchema["type"])
	}

	ageSchema := props["age"].(map[string]any)
	if ageSchema["type"] != "integer" {
		t.Errorf("age type = %v, want integer", ageSchema["type"])
	}

	required, ok := s["required"].([]any)
	if !ok {
		t.Fatal("required not a slice")
	}

	requiredSet := make(map[string]bool)
	for _, r := range required {
		requiredSet[r.(string)] = true
	}

	for _, field := range []string{"name", "age", "email"} {
		if !requiredSet[field] {
			t.Errorf("field %q should be required", field)
		}
	}

	if s["additionalProperties"] != false {
		t.Error("additionalProperties should be false")
	}
}

type nestedStruct struct {
	Title   string `json:"title"`
	Address struct {
		Street string `json:"street"`
		City   string `json:"city"`
	} `json:"address"`
}

func TestGenerateNestedStruct(t *testing.T) {
	raw := schema.Generate[nestedStruct]()

	var s map[string]any
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	props := s["properties"].(map[string]any)
	addrSchema := props["address"].(map[string]any)

	if addrSchema["type"] != "object" {
		t.Errorf("address type = %v, want object", addrSchema["type"])
	}

	addrProps := addrSchema["properties"].(map[string]any)
	if len(addrProps) != 2 {
		t.Errorf("address properties count = %d, want 2", len(addrProps))
	}
}

type sliceStruct struct {
	Tags []string `json:"tags"`
	IDs  []int    `json:"ids"`
}

func TestGenerateSliceFields(t *testing.T) {
	raw := schema.Generate[sliceStruct]()

	var s map[string]any
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	props := s["properties"].(map[string]any)

	tagsSchema := props["tags"].(map[string]any)
	if tagsSchema["type"] != "array" {
		t.Errorf("tags type = %v, want array", tagsSchema["type"])
	}

	items := tagsSchema["items"].(map[string]any)
	if items["type"] != "string" {
		t.Errorf("tags items type = %v, want string", items["type"])
	}

	idsSchema := props["ids"].(map[string]any)
	idsItems := idsSchema["items"].(map[string]any)
	if idsItems["type"] != "integer" {
		t.Errorf("ids items type = %v, want integer", idsItems["type"])
	}
}

type pointerStruct struct {
	Name     string  `json:"name"`
	Nickname *string `json:"nickname"`
}

func TestGeneratePointerFieldNullable(t *testing.T) {
	raw := schema.Generate[pointerStruct]()

	var s map[string]any
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	props := s["properties"].(map[string]any)

	nickSchema := props["nickname"].(map[string]any)
	if nickSchema["nullable"] != true {
		t.Error("pointer field should have nullable: true")
	}

	// Pointer fields should not be in required.
	required, _ := s["required"].([]any)
	for _, r := range required {
		if r.(string) == "nickname" {
			t.Error("pointer field nickname should not be required")
		}
	}
}

type optionalStruct struct {
	Name    string `json:"name"`
	Alias   string `json:"alias,omitempty"`
	Default string `json:"default,omitzero"`
}

func TestGenerateOmitemptyOmitzero(t *testing.T) {
	raw := schema.Generate[optionalStruct]()

	var s map[string]any
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	required, ok := s["required"].([]any)
	if !ok {
		t.Fatal("required not a slice")
	}

	requiredSet := make(map[string]bool)
	for _, r := range required {
		requiredSet[r.(string)] = true
	}

	if !requiredSet["name"] {
		t.Error("name should be required")
	}

	if requiredSet["alias"] {
		t.Error("alias (omitempty) should not be required")
	}

	if requiredSet["default"] {
		t.Error("default (omitzero) should not be required")
	}
}

type taggedStruct struct {
	Status string `json:"status" jsonschema:"description=The current status,enum=active|inactive|pending"`
	Name   string `json:"name" jsonschema:"description=The user name"`
}

func TestGenerateJSONSchemaTags(t *testing.T) {
	raw := schema.Generate[taggedStruct]()

	var s map[string]any
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	props := s["properties"].(map[string]any)

	statusSchema := props["status"].(map[string]any)
	if statusSchema["description"] != "The current status" {
		t.Errorf("status description = %v, want 'The current status'", statusSchema["description"])
	}

	enumVals, ok := statusSchema["enum"].([]any)
	if !ok {
		t.Fatal("status enum not a slice")
	}

	if len(enumVals) != 3 {
		t.Errorf("enum length = %d, want 3", len(enumVals))
	}

	expected := []string{"active", "inactive", "pending"}
	for i, v := range enumVals {
		if v.(string) != expected[i] {
			t.Errorf("enum[%d] = %v, want %v", i, v, expected[i])
		}
	}

	nameSchema := props["name"].(map[string]any)
	if nameSchema["description"] != "The user name" {
		t.Errorf("name description = %v, want 'The user name'", nameSchema["description"])
	}
}

type timeStruct struct {
	CreatedAt time.Time `json:"created_at"`
}

func TestGenerateTimeType(t *testing.T) {
	raw := schema.Generate[timeStruct]()

	var s map[string]any
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	props := s["properties"].(map[string]any)
	createdSchema := props["created_at"].(map[string]any)

	if createdSchema["type"] != "string" {
		t.Errorf("time type = %v, want string", createdSchema["type"])
	}

	if createdSchema["format"] != "date-time" {
		t.Errorf("time format = %v, want date-time", createdSchema["format"])
	}
}

type boolFloatStruct struct {
	Active bool    `json:"active"`
	Score  float64 `json:"score"`
}

func TestGenerateBoolAndFloat(t *testing.T) {
	raw := schema.Generate[boolFloatStruct]()

	var s map[string]any
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	props := s["properties"].(map[string]any)

	activeSchema := props["active"].(map[string]any)
	if activeSchema["type"] != "boolean" {
		t.Errorf("bool type = %v, want boolean", activeSchema["type"])
	}

	scoreSchema := props["score"].(map[string]any)
	if scoreSchema["type"] != "number" {
		t.Errorf("float type = %v, want number", scoreSchema["type"])
	}
}

type jsonIgnoredStruct struct {
	Name   string `json:"name"`
	Secret string `json:"-"`
}

func TestGenerateJSONIgnored(t *testing.T) {
	raw := schema.Generate[jsonIgnoredStruct]()

	var s map[string]any
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	props := s["properties"].(map[string]any)

	if _, exists := props["Secret"]; exists {
		t.Error("json:\"-\" field should be excluded from schema")
	}

	if len(props) != 1 {
		t.Errorf("properties count = %d, want 1", len(props))
	}
}

type unexportedStruct struct {
	Name string `json:"name"`
	age  int    //nolint:unused
}

func TestGenerateUnexportedFieldsIgnored(t *testing.T) {
	raw := schema.Generate[unexportedStruct]()

	var s map[string]any
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	props := s["properties"].(map[string]any)

	if len(props) != 1 {
		t.Errorf("properties count = %d, want 1 (unexported should be ignored)", len(props))
	}
}
