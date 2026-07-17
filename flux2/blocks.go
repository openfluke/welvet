package flux2

// DoubleStreamBlock is one Flux2TransformerBlock (separate img/txt streams, joint attn).
type DoubleStreamBlock struct {
	ToQ, ToK, ToV             *Linear
	AddQ, AddK, AddV          *Linear
	ToOut, ToAddOut           *Linear
	NormQ, NormK              []float32 // headDim
	NormAddedQ, NormAddedK    []float32
	FFIn, FFOut               *Linear
	FFContextIn, FFContextOut *Linear
	Heads, HeadDim, Dim       int
	MLPHidden                 int
	Eps                       float64
}

// Forward runs one double-stream block.
// hidden / encoder are [seq*dim]; tembModImg/Txt are flat length dim*6 (2 mod sets).
func (b *DoubleStreamBlock) Forward(
	hidden, encoder []float32,
	imgSeq, txtSeq int,
	tembModImg, tembModTxt []float32,
	rope RotaryEmb,
) error {
	dim := b.Dim
	inner := b.Heads * b.HeadDim
	modsImg := SplitModulation(tembModImg, dim, 2)
	modsTxt := SplitModulation(tembModTxt, dim, 2)
	shiftMSA, scaleMSA, gateMSA := modsImg[0].Shift, modsImg[0].Scale, modsImg[0].Gate
	shiftMLP, scaleMLP, gateMLP := modsImg[1].Shift, modsImg[1].Scale, modsImg[1].Gate
	cShiftMSA, cScaleMSA, cGateMSA := modsTxt[0].Shift, modsTxt[0].Scale, modsTxt[0].Gate
	cShiftMLP, cScaleMLP, cGateMLP := modsTxt[1].Shift, modsTxt[1].Scale, modsTxt[1].Gate

	normImg := make([]float32, imgSeq*dim)
	copy(normImg, hidden[:imgSeq*dim])
	LayerNormNoAffineSeq(normImg, imgSeq, dim, b.Eps)
	ApplyModulate(normImg, shiftMSA, scaleMSA, imgSeq, dim)

	normTxt := make([]float32, txtSeq*dim)
	copy(normTxt, encoder[:txtSeq*dim])
	LayerNormNoAffineSeq(normTxt, txtSeq, dim, b.Eps)
	ApplyModulate(normTxt, cShiftMSA, cScaleMSA, txtSeq, dim)

	qImg := make([]float32, imgSeq*inner)
	kImg := make([]float32, imgSeq*inner)
	vImg := make([]float32, imgSeq*inner)
	qTxt := make([]float32, txtSeq*inner)
	kTxt := make([]float32, txtSeq*inner)
	vTxt := make([]float32, txtSeq*inner)
	if err := b.ToQ.MatMulSeq(normImg, qImg, imgSeq); err != nil {
		return err
	}
	if err := b.ToK.MatMulSeq(normImg, kImg, imgSeq); err != nil {
		return err
	}
	if err := b.ToV.MatMulSeq(normImg, vImg, imgSeq); err != nil {
		return err
	}
	if err := b.AddQ.MatMulSeq(normTxt, qTxt, txtSeq); err != nil {
		return err
	}
	if err := b.AddK.MatMulSeq(normTxt, kTxt, txtSeq); err != nil {
		return err
	}
	if err := b.AddV.MatMulSeq(normTxt, vTxt, txtSeq); err != nil {
		return err
	}

	attnImg := make([]float32, imgSeq*inner)
	attnTxt := make([]float32, txtSeq*inner)
	JointAttention(qImg, kImg, vImg, qTxt, kTxt, vTxt,
		b.NormQ, b.NormK, b.NormAddedQ, b.NormAddedK,
		rope, imgSeq, txtSeq, b.Heads, b.HeadDim, b.Eps,
		attnImg, attnTxt)

	outImg := make([]float32, imgSeq*dim)
	outTxt := make([]float32, txtSeq*dim)
	if err := b.ToOut.MatMulSeq(attnImg, outImg, imgSeq); err != nil {
		return err
	}
	if err := b.ToAddOut.MatMulSeq(attnTxt, outTxt, txtSeq); err != nil {
		return err
	}

	ApplyGateResidual(hidden, outImg, gateMSA, imgSeq, dim)

	normImg2 := make([]float32, imgSeq*dim)
	copy(normImg2, hidden[:imgSeq*dim])
	LayerNormNoAffineSeq(normImg2, imgSeq, dim, b.Eps)
	ApplyModulate(normImg2, shiftMLP, scaleMLP, imgSeq, dim)
	ff := make([]float32, imgSeq*dim)
	if err := swigluFF(b.FFIn, b.FFOut, normImg2, ff, imgSeq, dim, b.MLPHidden); err != nil {
		return err
	}
	ApplyGateResidual(hidden, ff, gateMLP, imgSeq, dim)

	ApplyGateResidual(encoder, outTxt, cGateMSA, txtSeq, dim)

	normTxt2 := make([]float32, txtSeq*dim)
	copy(normTxt2, encoder[:txtSeq*dim])
	LayerNormNoAffineSeq(normTxt2, txtSeq, dim, b.Eps)
	ApplyModulate(normTxt2, cShiftMLP, cScaleMLP, txtSeq, dim)
	ffCtx := make([]float32, txtSeq*dim)
	if err := swigluFF(b.FFContextIn, b.FFContextOut, normTxt2, ffCtx, txtSeq, dim, b.MLPHidden); err != nil {
		return err
	}
	ApplyGateResidual(encoder, ffCtx, cGateMLP, txtSeq, dim)
	return nil
}

func swigluFF(linIn, linOut *Linear, x, y []float32, seq, dim, mlpHidden int) error {
	// linear_in: dim → 2*mlpHidden, then SwiGLU, then linear_out: mlpHidden → dim
	tmp := make([]float32, seq*2*mlpHidden)
	if err := linIn.MatMulSeq(x, tmp, seq); err != nil {
		return err
	}
	gated := make([]float32, seq*mlpHidden)
	for s := 0; s < seq; s++ {
		off := s * 2 * mlpHidden
		gate := tmp[off : off+mlpHidden]
		up := tmp[off+mlpHidden : off+2*mlpHidden]
		SiLU(gate)
		dst := gated[s*mlpHidden : (s+1)*mlpHidden]
		for i := 0; i < mlpHidden; i++ {
			dst[i] = gate[i] * up[i]
		}
	}
	return linOut.MatMulSeq(gated, y, seq)
}

// SingleStreamBlock is one Flux2SingleTransformerBlock (parallel attn+MLP).
type SingleStreamBlock struct {
	ToQKVMLP *Linear // out = 3*inner + 2*mlpHidden
	ToOut    *Linear // in = inner + mlpHidden, out = dim
	NormQ    []float32
	NormK    []float32
	Heads    int
	HeadDim  int
	Dim      int
	MLPHidden int
	Eps      float64
}

// Forward runs one single-stream block on concatenated txt||img hidden [seq*dim].
func (b *SingleStreamBlock) Forward(hidden []float32, seq int, tembMod []float32, rope RotaryEmb) error {
	dim := b.Dim
	inner := b.Heads * b.HeadDim
	mods := SplitModulation(tembMod, dim, 1)
	shift, scale, gate := mods[0].Shift, mods[0].Scale, mods[0].Gate

	normed := make([]float32, seq*dim)
	copy(normed, hidden[:seq*dim])
	LayerNormNoAffineSeq(normed, seq, dim, b.Eps)
	ApplyModulate(normed, shift, scale, seq, dim)

	fusedOut := 3*inner + 2*b.MLPHidden
	fused := make([]float32, seq*fusedOut)
	if err := b.ToQKVMLP.MatMulSeq(normed, fused, seq); err != nil {
		return err
	}

	q := make([]float32, seq*inner)
	k := make([]float32, seq*inner)
	v := make([]float32, seq*inner)
	mlpIn := make([]float32, seq*2*b.MLPHidden)
	for s := 0; s < seq; s++ {
		src := fused[s*fusedOut : (s+1)*fusedOut]
		copy(q[s*inner:(s+1)*inner], src[0:inner])
		copy(k[s*inner:(s+1)*inner], src[inner:2*inner])
		copy(v[s*inner:(s+1)*inner], src[2*inner:3*inner])
		copy(mlpIn[s*2*b.MLPHidden:(s+1)*2*b.MLPHidden], src[3*inner:])
	}

	attnOut := make([]float32, seq*inner)
	SelfAttention(q, k, v, b.NormQ, b.NormK, rope, seq, b.Heads, b.HeadDim, b.Eps, attnOut)

	mlpHidden := make([]float32, seq*b.MLPHidden)
	for s := 0; s < seq; s++ {
		off := s * 2 * b.MLPHidden
		gateA := mlpIn[off : off+b.MLPHidden]
		up := mlpIn[off+b.MLPHidden : off+2*b.MLPHidden]
		SiLU(gateA)
		dst := mlpHidden[s*b.MLPHidden : (s+1)*b.MLPHidden]
		for i := 0; i < b.MLPHidden; i++ {
			dst[i] = gateA[i] * up[i]
		}
	}

	// cat [attn | mlp] → to_out
	cat := make([]float32, seq*(inner+b.MLPHidden))
	for s := 0; s < seq; s++ {
		dst := cat[s*(inner+b.MLPHidden) : (s+1)*(inner+b.MLPHidden)]
		copy(dst[:inner], attnOut[s*inner:(s+1)*inner])
		copy(dst[inner:], mlpHidden[s*b.MLPHidden:(s+1)*b.MLPHidden])
	}
	proj := make([]float32, seq*dim)
	if err := b.ToOut.MatMulSeq(cat, proj, seq); err != nil {
		return err
	}
	ApplyGateResidual(hidden, proj, gate, seq, dim)
	return nil
}
