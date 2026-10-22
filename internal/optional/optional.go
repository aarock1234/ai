// Package optional reads optional settings that are represented as
// pointers, such as provider options whose zero value differs from their
// default.
package optional

// Or returns the value that value points to, or fallback when value is nil.
func Or[T any](value *T, fallback T) T {
	if value == nil {
		return fallback
	}

	return *value
}
