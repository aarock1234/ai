package schema

import "encoding/json"

// node is the subset of JSON Schema that Generate produces.
type node struct {
	Type                 string                `json:"type,omitempty"`
	Format               string                `json:"format,omitempty"`
	ContentEncoding      string                `json:"contentEncoding,omitempty"`
	Description          *string               `json:"description,omitempty"`
	Enum                 []string              `json:"enum,omitempty"`
	AnyOf                []*node               `json:"anyOf,omitempty"`
	Items                *node                 `json:"items,omitempty"`
	MinItems             *int                  `json:"minItems,omitempty"`
	MaxItems             *int                  `json:"maxItems,omitempty"`
	Properties           *map[string]*node     `json:"properties,omitempty"`
	Required             []string              `json:"required,omitempty"`
	AdditionalProperties *additionalProperties `json:"additionalProperties,omitempty"`

	// nullable adds "null" to the type, which is then written as a list.
	nullable bool
}

// plainNode marshals a node without its MarshalJSON method.
type plainNode node

// nullableNode replaces the type of a node with a type list.
type nullableNode struct {
	plainNode

	Type []string `json:"type"`
}

// MarshalJSON writes the node, with the type list [type, "null"] when it is
// nullable.
func (n node) MarshalJSON() ([]byte, error) {
	if !n.nullable {
		return json.Marshal(plainNode(n))
	}

	return json.Marshal(nullableNode{
		plainNode: plainNode(n),
		Type: []string{
			n.Type,
			"null",
		},
	})
}

// additionalProperties is either a boolean or a schema.
type additionalProperties struct {
	allowed *bool
	schema  *node
}

// MarshalJSON writes the boolean when set, otherwise the schema.
func (p additionalProperties) MarshalJSON() ([]byte, error) {
	if p.allowed != nil {
		return json.Marshal(*p.allowed)
	}

	return json.Marshal(p.schema)
}
