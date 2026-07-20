package seed

// FindLayerSeed searches for a layer_seed whose He-init matches master exactly.
func FindLayerSeed(master []float32, inputSize int, hints ...uint64) (uint64, bool) {
	return FindLayerSeedForWeights(master, inputSize, hints...)
}

// FindLayerSeedForWeights is the loom alias.
func FindLayerSeedForWeights(master []float32, inputSize int, hints ...uint64) (uint64, bool) {
	if len(master) == 0 {
		return 0, false
	}
	if inputSize <= 0 {
		inputSize = 1
	}
	try := make([]uint64, 0, len(hints)+4)
	try = append(try, hints...)
	try = append(try, From("invert-hint", len(master), FingerprintF32(master)))
	best := try[0]
	bestMiss := mismatches(master, inputSize, best)
	if bestMiss == 0 {
		return best, true
	}
	rng := New(From("invert-search", best, uint64(len(master))))
	const trials = 50_000
	for t := 0; t < trials; t++ {
		candidate := best
		if t%50_000 == 0 && t > 0 {
			candidate = rng.Uint64()
		} else {
			candidate = mutate(candidate, rng.Uint64())
		}
		miss := mismatches(master, inputSize, candidate)
		if miss < bestMiss {
			bestMiss = miss
			best = candidate
			if bestMiss == 0 {
				return best, true
			}
		}
	}
	return best, bestMiss == 0
}

func mismatches(master []float32, inputSize int, seed uint64) int {
	tmp := make([]float32, len(master))
	InitFloat32He(tmp, inputSize, seed)
	miss := 0
	for i := range master {
		if master[i] != tmp[i] {
			miss++
		}
	}
	return miss
}

func mutate(seed, noise uint64) uint64 {
	switch noise % 3 {
	case 0:
		bit := (noise / 3) % 64
		return seed ^ (1 << bit)
	case 1:
		return seed + (noise >> 6) + 1
	default:
		return seed ^ noise
	}
}
