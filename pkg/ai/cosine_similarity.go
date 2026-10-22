package ai

import "math"

// vectorLengths is the Value of the InvalidArgumentError that reports vectors
// of different lengths.
type vectorLengths struct {
	Vector1Length int `json:"vector1Length"`
	Vector2Length int `json:"vector2Length"`
}

// CosineSimilarity returns the cosine of the angle between two vectors, such
// as two embeddings: 1 for the same direction, 0 for orthogonal vectors and
// -1 for opposite directions. It returns 0 when either vector is empty or the
// zero vector, and an InvalidArgumentError when the lengths differ.
func CosineSimilarity(vector1 []float64, vector2 []float64) (float64, error) {
	if len(vector1) != len(vector2) {
		return 0, &InvalidArgumentError{
			Parameter: "vector1,vector2",
			Value: vectorLengths{
				Vector1Length: len(vector1),
				Vector2Length: len(vector2),
			},
			Message: "Vectors must have the same length",
		}
	}

	var magnitudeSquared1 float64
	var magnitudeSquared2 float64
	var dotProduct float64

	for index, value1 := range vector1 {
		value2 := vector2[index]

		magnitudeSquared1 += value1 * value1
		magnitudeSquared2 += value2 * value2
		dotProduct += value1 * value2
	}

	if magnitudeSquared1 == 0 || magnitudeSquared2 == 0 {
		return 0, nil
	}

	magnitudes := math.Sqrt(magnitudeSquared1) * math.Sqrt(magnitudeSquared2)

	return dotProduct / magnitudes, nil
}
