package fountain

import (
	"fmt"
	"math"

	"github.com/openfluke/welvet/stub/seed"
)

// RecoverWeightBlobs LT-sprays/peels equal-sized source blocks under a lossy channel.
func RecoverWeightBlobs(blobs [][]byte, spraySeed uint64, loss, maxOverhead float64) (recovered [][]byte, received, sprayed int, err error) {
	enc, err := NewLTEncoder(blobs, spraySeed)
	if err != nil {
		return nil, 0, 0, err
	}
	dec := NewLTDecoder(len(blobs), len(blobs[0]))
	lossRng := seed.New(seed.From("loom-poly-neural-fountain-loss", spraySeed))
	if maxOverhead < 1 {
		maxOverhead = 1
	}
	maxSpray := int(math.Ceil(float64(len(blobs)) * (1 + maxOverhead)))
	if floor := len(blobs) * 8; maxSpray < floor {
		maxSpray = floor
	}
	last := 0
	stalls := 0
	for !dec.Done() && sprayed < maxSpray {
		drop := enc.Spray()
		sprayed++
		if lossRng.Float64() < loss {
			continue
		}
		received++
		dec.Catch(drop)
		known := dec.KnownCount()
		if known == last {
			stalls++
			if stalls%50 == 0 {
				dec.TryResidualGE(maxGEUnknowns)
			}
		} else {
			stalls = 0
			last = known
		}
	}
	if !dec.Done() {
		dec.TryResidualGE(maxGEUnknowns)
	}
	extra := 0
	for !dec.Done() && extra < len(blobs)*20 {
		drop := enc.Spray()
		sprayed++
		extra++
		if lossRng.Float64() < loss {
			continue
		}
		received++
		dec.Catch(drop)
		dec.TryResidualGE(maxGEUnknowns)
	}
	if !dec.Done() {
		return nil, received, sprayed, fmt.Errorf("fountain: stalled at %d/%d (recv=%d sprayed=%d)",
			dec.KnownCount(), len(blobs), received, sprayed)
	}
	if !BlocksEqual(blobs, dec.Recovered) {
		return nil, received, sprayed, fmt.Errorf("fountain: recovered != sources")
	}
	return dec.Recovered, received, sprayed, nil
}
