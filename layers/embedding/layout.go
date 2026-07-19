package embedding

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

type layout struct {
	batch  int
	seq    int
	emb    int
	vocab  int
	nTok   int
	inRank int
	inLast int // last dim of input when rank==3 (1 or emb for chain)
}

func parseLayout[T core.Numeric](cfg Config, input *core.Tensor[T]) (layout, error) {
	var lay layout
	lay.emb = cfg.EmbeddingDim
	lay.vocab = cfg.VocabSize
	if input == nil || len(input.Data) == 0 {
		return lay, fmt.Errorf("embedding: empty input")
	}
	switch len(input.Shape) {
	case 1:
		lay.batch = 1
		lay.seq = input.Shape[0]
		lay.inRank = 1
		lay.inLast = 1
	case 2:
		lay.batch = input.Shape[0]
		lay.seq = input.Shape[1]
		lay.inRank = 2
		lay.inLast = 1
	case 3:
		// [B,T,1] tokens, or [B,T,E] after a prior embedding (chain: use channel 0 as ID).
		lay.batch = input.Shape[0]
		lay.seq = input.Shape[1]
		lay.inRank = 3
		lay.inLast = input.Shape[2]
		if lay.inLast != 1 && lay.inLast != cfg.EmbeddingDim {
			return lay, fmt.Errorf("embedding: input last dim %d want 1 or emb=%d", lay.inLast, cfg.EmbeddingDim)
		}
	default:
		return lay, fmt.Errorf("embedding: input rank %d want 1..3", len(input.Shape))
	}
	if lay.batch <= 0 || lay.seq <= 0 {
		return lay, fmt.Errorf("embedding: bad batch/seq")
	}
	lay.nTok = lay.batch * lay.seq
	want := lay.nTok * lay.inLast
	if input.Len() < want {
		return lay, fmt.Errorf("embedding: input short len=%d want=%d", input.Len(), want)
	}
	return lay, nil
}

func tokenID[T core.Numeric](input *core.Tensor[T], lay layout, tok int) int {
	var v float64
	if lay.inRank == 3 && lay.inLast > 1 {
		v = core.AsFloat64(input.Data[tok*lay.inLast])
	} else if lay.inRank == 3 {
		v = core.AsFloat64(input.Data[tok])
	} else {
		v = core.AsFloat64(input.Data[tok])
	}
	id := int(v)
	if id < 0 {
		id = -id
	}
	if lay.vocab > 0 {
		id = id % lay.vocab
	}
	return id
}

func outShape(lay layout) []int {
	if lay.inRank == 1 {
		return []int{lay.seq, lay.emb}
	}
	return []int{lay.batch, lay.seq, lay.emb}
}
