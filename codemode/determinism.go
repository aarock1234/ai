package codemode

import (
	"strconv"
	"time"
)

// Seeds upstream's runtime falls back to for a malformed seed half.
const (
	defaultRandomState0 = 0x243f6a8885a308d3
	defaultRandomState1 = 0x13198a2e03707344
	zeroSeedRandomState = 0x9e3779b97f4a7c15
)

// runRandom is the Math.random of upstream's runtime: xorshift128+ seeded
// from the continuation's 128-bit hexadecimal seed, so that a replayed
// program draws the same numbers in either implementation.
type runRandom struct {
	state0 uint64
	state1 uint64
}

func newRunRandom(seed string) *runRandom {
	random := &runRandom{
		state0: seedHalf(seed, 0, defaultRandomState0),
		state1: seedHalf(seed, 16, defaultRandomState1),
	}

	if random.state0|random.state1 == 0 {
		random.state1 = zeroSeedRandomState
	}

	return random
}

func seedHalf(seed string, offset int, fallback uint64) uint64 {
	if len(seed) < offset+16 {
		return fallback
	}

	value, err := strconv.ParseUint(seed[offset:offset+16], 16, 64)
	if err != nil {
		return fallback
	}

	return value
}

// next returns a number in [0, 1) with 53 random bits.
func (r *runRandom) next() float64 {
	s1 := r.state0
	s0 := r.state1

	r.state0 = s0
	s1 ^= s1 << 23
	r.state1 = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26)

	return float64((r.state1+s0)>>11) / (1 << 53)
}

// runClock is the clock of upstream's runtime: every read returns the
// current millisecond and advances it by one, and each host function
// outcome resets it to the host clock when the call settled.
type runClock struct {
	nowMs int64
}

func (c *runClock) now() time.Time {
	current := c.nowMs
	c.nowMs++

	return time.UnixMilli(current)
}

func (c *runClock) reset(nowMs int64) {
	c.nowMs = nowMs
}
