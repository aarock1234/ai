package ai

import (
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"math"

	"github.com/aarock1234/ai/pkg/model"
)

// pcm16BytesPerSample is the size of one 16-bit PCM sample.
const pcm16BytesPerSample = 2

// EncodeRealtimeAudio converts samples between -1 and 1 into base64
// little-endian 16-bit PCM, the audio format of realtime input events.
// Samples outside the range are clamped.
func EncodeRealtimeAudio(samples []float32) string {
	pcm := make([]byte, len(samples)*pcm16BytesPerSample)

	for index, sample := range samples {
		binary.LittleEndian.PutUint16(pcm[index*pcm16BytesPerSample:], uint16(pcm16Sample(sample)))
	}

	return base64.StdEncoding.EncodeToString(pcm)
}

// pcm16Sample scales a clamped sample asymmetrically, so that -1 maps to
// the lowest 16-bit value and 1 to the highest.
func pcm16Sample(sample float32) int16 {
	clamped := math.Max(-1, math.Min(1, float64(sample)))

	if clamped < 0 {
		return int16(clamped * 0x8000)
	}

	return int16(clamped * 0x7fff)
}

// DecodeRealtimeAudio converts base64 little-endian 16-bit PCM, the audio
// format of realtime output events, into samples between -1 and 1.
func DecodeRealtimeAudio(audio string) ([]float32, error) {
	pcm, err := base64.StdEncoding.DecodeString(audio)
	if err != nil {
		return nil, fmt.Errorf("%w: decode realtime audio: %w", model.ErrInvalidArgument, err)
	}

	if len(pcm)%pcm16BytesPerSample != 0 {
		return nil, fmt.Errorf("%w: realtime audio has an odd number of bytes", model.ErrInvalidArgument)
	}

	samples := make([]float32, len(pcm)/pcm16BytesPerSample)
	for index := range samples {
		sample := int16(binary.LittleEndian.Uint16(pcm[index*pcm16BytesPerSample:]))
		samples[index] = float32(sample) / 32768
	}

	return samples, nil
}

// ResampleAudio converts samples from inputRate to outputRate in Hz by
// linear interpolation, which suits voice audio. Equal rates return the
// input unchanged.
func ResampleAudio(samples []float32, inputRate, outputRate int) []float32 {
	if inputRate == outputRate || len(samples) == 0 || inputRate <= 0 || outputRate <= 0 {
		return samples
	}

	ratio := float64(inputRate) / float64(outputRate)
	output := make([]float32, int(math.Round(float64(len(samples))/ratio)))

	for index := range output {
		position := float64(index) * ratio
		floor := min(int(position), len(samples)-1)
		ceiling := min(floor+1, len(samples)-1)
		fraction := float32(position - float64(floor))

		output[index] = samples[floor]*(1-fraction) + samples[ceiling]*fraction
	}

	return output
}
