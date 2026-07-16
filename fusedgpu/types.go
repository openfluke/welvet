package fusedgpu

type q4Mat struct {
	rows, cols int
	scales     []float32
	packed     []uint32
}

type rmsW struct {
	w []float32
}

type blockCPU struct {
	attnNorm rmsW
	q, k, v  q4Mat
	o        q4Mat
	mlpNorm  rmsW
	gate, up q4Mat
	down     q4Mat
}

type modelCPU struct {
	hidden, vocab, layers int
	heads, kvHeads        int
	headDim, qDim, kvDim  int
	intermediate          int
	eps                   float32
	ropeTheta             float32
	maxSeq                int

	embed     []float32
	finalNorm []float32
	lmScales  []float32
	lmPacked  []uint32
	blocks    []blockCPU
}
