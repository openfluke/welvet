package seed

import "math"

// RNG is a deterministic pseudo-random generator built from xorshift64*
// (Vigna, 2014). Same seed always produces the same sequence — no math/rand.
type RNG struct {
	state uint64
}

// New returns an RNG initialized from seed. Zero is mapped to a fixed nonzero
// state so the generator still advances.
func New(seed uint64) *RNG {
	if seed == 0 {
		seed = 0xdeadbeefcafebabe
	}
	return &RNG{state: splitmix64(seed)}
}

// Seed returns the RNG to the start of the sequence for seed.
func (r *RNG) Seed(seed uint64) {
	if seed == 0 {
		seed = 0xdeadbeefcafebabe
	}
	r.state = splitmix64(seed)
}

// Uint64 returns the next 64-bit value in the sequence.
func (r *RNG) Uint64() uint64 {
	x := r.state
	x ^= x >> 12
	x ^= x << 25
	x ^= x >> 27
	r.state = x
	return x * 0x2545F4914F6CDD1D
}

// Float64 returns a value in [0, 1) with 53 bits of precision.
func (r *RNG) Float64() float64 {
	return float64(r.Uint64()>>11) / (1 << 53)
}

// Intn returns a uniform int in [0, n). Panics if n <= 0.
func (r *RNG) Intn(n int) int {
	if n <= 0 {
		panic("seed: Intn non-positive")
	}
	return int(r.Uint64() % uint64(n))
}

// NormFloat64 returns a standard normal sample (mean 0, stddev 1).
func (r *RNG) NormFloat64() float64 {
	for {
		u1 := r.Float64()*2 - 1
		u2 := r.Float64()*2 - 1
		s := u1*u1 + u2*u2
		if s > 0 && s < 1 {
			mul := math.Sqrt(-2 * math.Log(s) / s)
			return u1 * mul
		}
	}
}

// splitmix64 scrambles a 64-bit value (used for seed expansion).
func splitmix64(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	z := x
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// NewSeedRNG is the loom/poly alias for New.
func NewSeedRNG(seed uint64) *RNG { return New(seed) }
