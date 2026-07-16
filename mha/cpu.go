package mha

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/dense"
)

func prepareKV(l *Layer, lay layout, msl, kvDim int) {
	incremental := l.Cfg.Mode == ModeSelf &&
		lay.batch == 1 && lay.seqLen == 1 && l.KVCacheK != nil && l.KVOffset > 0 &&
		(l.Cfg.Mask == MaskCausal || l.Cfg.Mask == MaskSlidingWindow || l.Cfg.Mask == MaskPrefixLM)
	if incremental {
		return
	}
	l.KVCacheK = make([]float64, msl*kvDim)
	l.KVCacheV = make([]float64, msl*kvDim)
	l.KVOffset = 0
}

// ForwardCPUTiled — projections via dense; mask/pos/mode policy in f64 attn ALU.
func ForwardCPUTiled[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	return forwardHost(l, input)
}

// BackwardCPUTiled — reverse of forward; gradW is concat(Q,K,V,O) matrices.
func BackwardCPUTiled[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	return backwardHost(l, gradOut, input, pre)
}

func forwardHost[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	cfg := l.Cfg
	allow, err := l.allowFn()
	if err != nil {
		return nil, nil, err
	}
	lay, err := parseLayout(cfg.DModel, input)
	if err != nil {
		return nil, nil, err
	}
	qDim, kvDim := cfg.QDim(), cfg.KVDim()
	msl := cfg.MaxSeqLen
	d := cfg.DModel

	flatIn := flattenTokens(input, lay)
	_, qPost, err := dense.Forward(l.Q, flatIn)
	if err != nil {
		return nil, nil, fmt.Errorf("mha Q: %w", err)
	}

	var kPost, vPost *core.Tensor[T]
	var ctxLay layout
	kvLen := 0
	kvStart := 0

	if cfg.Mode == ModeCross {
		ctx, ok := l.Context.(*core.Tensor[T])
		if !ok || ctx == nil {
			return nil, nil, fmt.Errorf("mha: ModeCross requires SetContext / ForwardWithContext")
		}
		ctxLay, err = parseLayout(cfg.DModel, ctx)
		if err != nil {
			return nil, nil, fmt.Errorf("mha context: %w", err)
		}
		if ctxLay.batch != lay.batch {
			return nil, nil, fmt.Errorf("mha: context batch %d != input batch %d", ctxLay.batch, lay.batch)
		}
		flatCtx := flattenTokens(ctx, ctxLay)
		_, kPost, err = dense.Forward(l.K, flatCtx)
		if err != nil {
			return nil, nil, fmt.Errorf("mha K(ctx): %w", err)
		}
		_, vPost, err = dense.Forward(l.V, flatCtx)
		if err != nil {
			return nil, nil, fmt.Errorf("mha V(ctx): %w", err)
		}
		// Per-batch context fills cache slots 0..ctxSeq-1 (no incremental decode).
		need := ctxLay.seqLen
		if need > msl {
			return nil, nil, fmt.Errorf("mha: context seq %d > MaxSeqLen %d", need, msl)
		}
		l.KVCacheK = make([]float64, msl*kvDim)
		l.KVCacheV = make([]float64, msl*kvDim)
		l.KVOffset = 0
		kvLen = ctxLay.seqLen
		kvStart = 0
	} else {
		_, kPost, err = dense.Forward(l.K, flatIn)
		if err != nil {
			return nil, nil, fmt.Errorf("mha K: %w", err)
		}
		_, vPost, err = dense.Forward(l.V, flatIn)
		if err != nil {
			return nil, nil, fmt.Errorf("mha V: %w", err)
		}
		prepareKV(l, lay, msl, kvDim)
		kvStart = l.KVOffset
	}

	attnFlat := core.NewTensor[T](lay.batch*lay.seqLen, qDim)
	qGamma := l.QNormWeight
	kGamma := l.KNormWeight

	needScratch := lay.seqLen * qDim
	if needScratch > len(l.DecodeScratchQ) {
		l.DecodeScratchQ = make([]float64, needScratch)
	}
	if needScratch > len(l.DecodeScratchAttn) {
		l.DecodeScratchAttn = make([]float64, needScratch)
	}

	for b := 0; b < lay.batch; b++ {
		seqBase := kvStart
		if cfg.Mode == ModeSelf {
			seqBase = kvStart + b*lay.seqLen
		}
		Q := l.DecodeScratchQ[:lay.seqLen*qDim]

		if cfg.Mode == ModeCross {
			// Fill KV from this batch's context once.
			for s := 0; s < ctxLay.seqLen; s++ {
				tok := b*ctxLay.seqLen + s
				kRow := l.KVCacheK[s*kvDim : (s+1)*kvDim]
				vRow := l.KVCacheV[s*kvDim : (s+1)*kvDim]
				for i := 0; i < kvDim; i++ {
					kRow[i] = core.AsFloat64(kPost.Data[tok*kvDim+i])
					vRow[i] = core.AsFloat64(vPost.Data[tok*kvDim+i])
				}
				if cfg.QKNorm {
					applyQKNormInPlace(kRow, kGamma, cfg.NumKVHeads, cfg.HeadDim, cfg.QKNormEps)
				}
				if cfg.RoPEOnContext && cfg.UsesRoPE() {
					applyRoPE(kRow, s, cfg.NumKVHeads, cfg.HeadDim, cfg.RoPETheta)
				}
			}
			for s := 0; s < lay.seqLen; s++ {
				tok := b*lay.seqLen + s
				for i := 0; i < qDim; i++ {
					Q[s*qDim+i] = core.AsFloat64(qPost.Data[tok*qDim+i])
				}
				if cfg.QKNorm {
					applyQKNormInPlace(Q[s*qDim:(s+1)*qDim], qGamma, cfg.NumHeads, cfg.HeadDim, cfg.QKNormEps)
				}
				if cfg.UsesRoPE() {
					applyRoPE(Q[s*qDim:(s+1)*qDim], s, cfg.NumHeads, cfg.HeadDim, cfg.RoPETheta)
				}
			}
		} else {
			for s := 0; s < lay.seqLen; s++ {
				pos := seqBase + s
				tok := b*lay.seqLen + s
				for i := 0; i < qDim; i++ {
					Q[s*qDim+i] = core.AsFloat64(qPost.Data[tok*qDim+i])
				}
				kRow := l.KVCacheK[(pos%msl)*kvDim : (pos%msl+1)*kvDim]
				vRow := l.KVCacheV[(pos%msl)*kvDim : (pos%msl+1)*kvDim]
				for i := 0; i < kvDim; i++ {
					kRow[i] = core.AsFloat64(kPost.Data[tok*kvDim+i])
					vRow[i] = core.AsFloat64(vPost.Data[tok*kvDim+i])
				}
				if cfg.QKNorm {
					applyQKNormInPlace(Q[s*qDim:(s+1)*qDim], qGamma, cfg.NumHeads, cfg.HeadDim, cfg.QKNormEps)
					applyQKNormInPlace(kRow, kGamma, cfg.NumKVHeads, cfg.HeadDim, cfg.QKNormEps)
				}
				if cfg.UsesRoPE() {
					applyRoPE(Q[s*qDim:(s+1)*qDim], pos, cfg.NumHeads, cfg.HeadDim, cfg.RoPETheta)
					applyRoPE(kRow, pos, cfg.NumKVHeads, cfg.HeadDim, cfg.RoPETheta)
				}
			}
			kvLen = seqBase + lay.seqLen
		}

		attnOut := l.DecodeScratchAttn[:lay.seqLen*qDim]
		attnForward(cfg, attnOut, Q, l.KVCacheK, l.KVCacheV,
			lay.seqLen, seqBase, kvLen, msl, qDim, kvDim, allow)
		for s := 0; s < lay.seqLen; s++ {
			tok := b*lay.seqLen + s
			for i := 0; i < qDim; i++ {
				attnFlat.Data[tok*qDim+i] = core.FromFloat64[T](attnOut[s*qDim+i])
			}
		}
	}
	if cfg.Mode == ModeSelf {
		l.KVOffset = kvStart + lay.batch*lay.seqLen
	}

	_, oPost, err := dense.Forward(l.O, attnFlat)
	if err != nil {
		return nil, nil, fmt.Errorf("mha O: %w", err)
	}
	pre = reshapeSeq(attnFlat, lay, qDim)
	post = reshapeSeq(oPost, lay, d)
	return pre, post, nil
}

func backwardHost[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	cfg := l.Cfg
	allow, err := l.allowFn()
	if err != nil {
		return nil, nil, err
	}
	lay, err := parseLayout(cfg.DModel, input)
	if err != nil {
		return nil, nil, err
	}
	qDim, kvDim := cfg.QDim(), cfg.KVDim()
	msl := cfg.MaxSeqLen
	d := cfg.DModel
	bs := lay.batch * lay.seqLen

	flatIn := flattenTokens(input, lay)
	gradOutFlat := flattenTokens(gradOut, lay)
	if pre == nil || pre.Len() < bs*qDim {
		return nil, nil, fmt.Errorf("mha: pre (attn) missing or wrong size")
	}
	attnFlat := core.NewTensor[T](bs, qDim)
	copy(attnFlat.Data, pre.Data[:bs*qDim])

	oPre, _, err := dense.Forward(l.O, attnFlat)
	if err != nil {
		return nil, nil, fmt.Errorf("mha O recompute: %w", err)
	}
	gradAttn, gradWO, err := dense.Backward(l.O, gradOutFlat, attnFlat, oPre)
	if err != nil {
		return nil, nil, fmt.Errorf("mha O bwd: %w", err)
	}

	qPre, qPost, err := dense.Forward(l.Q, flatIn)
	if err != nil {
		return nil, nil, fmt.Errorf("mha Q recompute: %w", err)
	}

	var kPre, kPost, vPre, vPost *core.Tensor[T]
	var ctxLay layout
	var flatCtx *core.Tensor[T]
	kvLen, kvStart := 0, 0

	if cfg.Mode == ModeCross {
		ctx, ok := l.Context.(*core.Tensor[T])
		if !ok || ctx == nil {
			return nil, nil, fmt.Errorf("mha: ModeCross backward needs Context")
		}
		ctxLay, err = parseLayout(cfg.DModel, ctx)
		if err != nil {
			return nil, nil, err
		}
		flatCtx = flattenTokens(ctx, ctxLay)
		kPre, kPost, err = dense.Forward(l.K, flatCtx)
		if err != nil {
			return nil, nil, err
		}
		vPre, vPost, err = dense.Forward(l.V, flatCtx)
		if err != nil {
			return nil, nil, err
		}
		kvLen = ctxLay.seqLen
		kvStart = 0
	} else {
		kPre, kPost, err = dense.Forward(l.K, flatIn)
		if err != nil {
			return nil, nil, err
		}
		vPre, vPost, err = dense.Forward(l.V, flatIn)
		if err != nil {
			return nil, nil, err
		}
		kvEnd := l.KVOffset
		if kvEnd < bs {
			kvEnd = bs
		}
		kvStart = kvEnd - bs
	}

	gQAll := make([]float64, bs*qDim)
	gK := make([]float64, msl*kvDim)
	gV := make([]float64, msl*kvDim)
	qGamma, kGamma := l.QNormWeight, l.KNormWeight

	for b := 0; b < lay.batch; b++ {
		seqBase := kvStart
		if cfg.Mode == ModeSelf {
			seqBase = kvStart + b*lay.seqLen
		}
		cacheK := make([]float64, msl*kvDim)
		cacheV := make([]float64, msl*kvDim)
		Q := make([]float64, lay.seqLen*qDim)

		if cfg.Mode == ModeCross {
			for s := 0; s < ctxLay.seqLen; s++ {
				tok := b*ctxLay.seqLen + s
				kRow := cacheK[s*kvDim : (s+1)*kvDim]
				vRow := cacheV[s*kvDim : (s+1)*kvDim]
				for i := 0; i < kvDim; i++ {
					kRow[i] = core.AsFloat64(kPost.Data[tok*kvDim+i])
					vRow[i] = core.AsFloat64(vPost.Data[tok*kvDim+i])
				}
				if cfg.QKNorm {
					applyQKNormInPlace(kRow, kGamma, cfg.NumKVHeads, cfg.HeadDim, cfg.QKNormEps)
				}
				if cfg.RoPEOnContext && cfg.UsesRoPE() {
					applyRoPE(kRow, s, cfg.NumKVHeads, cfg.HeadDim, cfg.RoPETheta)
				}
			}
			for s := 0; s < lay.seqLen; s++ {
				tok := b*lay.seqLen + s
				for i := 0; i < qDim; i++ {
					Q[s*qDim+i] = core.AsFloat64(qPost.Data[tok*qDim+i])
				}
				if cfg.QKNorm {
					applyQKNormInPlace(Q[s*qDim:(s+1)*qDim], qGamma, cfg.NumHeads, cfg.HeadDim, cfg.QKNormEps)
				}
				if cfg.UsesRoPE() {
					applyRoPE(Q[s*qDim:(s+1)*qDim], s, cfg.NumHeads, cfg.HeadDim, cfg.RoPETheta)
				}
			}
		} else {
			for s := 0; s < lay.seqLen; s++ {
				pos := seqBase + s
				tok := b*lay.seqLen + s
				for i := 0; i < qDim; i++ {
					Q[s*qDim+i] = core.AsFloat64(qPost.Data[tok*qDim+i])
				}
				kRow := cacheK[(pos%msl)*kvDim : (pos%msl+1)*kvDim]
				vRow := cacheV[(pos%msl)*kvDim : (pos%msl+1)*kvDim]
				for i := 0; i < kvDim; i++ {
					kRow[i] = core.AsFloat64(kPost.Data[tok*kvDim+i])
					vRow[i] = core.AsFloat64(vPost.Data[tok*kvDim+i])
				}
				if cfg.QKNorm {
					applyQKNormInPlace(Q[s*qDim:(s+1)*qDim], qGamma, cfg.NumHeads, cfg.HeadDim, cfg.QKNormEps)
					applyQKNormInPlace(kRow, kGamma, cfg.NumKVHeads, cfg.HeadDim, cfg.QKNormEps)
				}
				if cfg.UsesRoPE() {
					applyRoPE(Q[s*qDim:(s+1)*qDim], pos, cfg.NumHeads, cfg.HeadDim, cfg.RoPETheta)
					applyRoPE(kRow, pos, cfg.NumKVHeads, cfg.HeadDim, cfg.RoPETheta)
				}
			}
			kvLen = seqBase + lay.seqLen
		}

		gradPre := make([]float64, lay.seqLen*qDim)
		for s := 0; s < lay.seqLen; s++ {
			tok := b*lay.seqLen + s
			for i := 0; i < qDim; i++ {
				gradPre[s*qDim+i] = core.AsFloat64(gradAttn.Data[tok*qDim+i])
			}
		}
		gQ := make([]float64, lay.seqLen*qDim)
		attnBackward(cfg, gradPre, Q, cacheK, cacheV, gQ, gK, gV,
			lay.seqLen, seqBase, kvLen, msl, qDim, kvDim, allow)

		for s := 0; s < lay.seqLen; s++ {
			pos := seqBase + s
			if cfg.Mode == ModeCross {
				pos = s
			}
			if cfg.UsesRoPE() {
				applyRoPEBackward(gQ[s*qDim:(s+1)*qDim], pos, cfg.NumHeads, cfg.HeadDim, cfg.RoPETheta)
			}
			tok := b*lay.seqLen + s
			copy(gQAll[tok*qDim:(tok+1)*qDim], gQ[s*qDim:(s+1)*qDim])
		}
		if cfg.Mode == ModeSelf && cfg.UsesRoPE() {
			for s := 0; s < lay.seqLen; s++ {
				pos := seqBase + s
				kIdx := pos % msl
				applyRoPEBackward(gK[kIdx*kvDim:(kIdx+1)*kvDim], pos, cfg.NumKVHeads, cfg.HeadDim, cfg.RoPETheta)
			}
		} else if cfg.Mode == ModeCross && cfg.RoPEOnContext && cfg.UsesRoPE() {
			for s := 0; s < ctxLay.seqLen; s++ {
				applyRoPEBackward(gK[s*kvDim:(s+1)*kvDim], s, cfg.NumKVHeads, cfg.HeadDim, cfg.RoPETheta)
			}
		}
	}

	gQT := core.NewTensor[T](bs, qDim)
	for i := 0; i < bs*qDim; i++ {
		gQT.Data[i] = core.FromFloat64[T](gQAll[i])
	}
	gradInQ, gradWQ, err := dense.Backward(l.Q, gQT, flatIn, qPre)
	if err != nil {
		return nil, nil, fmt.Errorf("mha Q bwd: %w", err)
	}

	var gradWK, gradWV *core.Tensor[T]
	var gradInK, gradInV *core.Tensor[T]

	if cfg.Mode == ModeCross {
		cbs := ctxLay.batch * ctxLay.seqLen
		gKT := core.NewTensor[T](cbs, kvDim)
		gVT := core.NewTensor[T](cbs, kvDim)
		for b := 0; b < lay.batch; b++ {
			for s := 0; s < ctxLay.seqLen; s++ {
				tok := b*ctxLay.seqLen + s
				for i := 0; i < kvDim; i++ {
					gKT.Data[tok*kvDim+i] = core.FromFloat64[T](gK[s*kvDim+i])
					gVT.Data[tok*kvDim+i] = core.FromFloat64[T](gV[s*kvDim+i])
				}
			}
		}
		gradInK, gradWK, err = dense.Backward(l.K, gKT, flatCtx, kPre)
		if err != nil {
			return nil, nil, fmt.Errorf("mha K bwd: %w", err)
		}
		gradInV, gradWV, err = dense.Backward(l.V, gVT, flatCtx, vPre)
		if err != nil {
			return nil, nil, fmt.Errorf("mha V bwd: %w", err)
		}
		_ = gradInK
		_ = gradInV
		gradIn = reshapeSeq(gradInQ, lay, d)
	} else {
		gKT := core.NewTensor[T](bs, kvDim)
		gVT := core.NewTensor[T](bs, kvDim)
		for b := 0; b < lay.batch; b++ {
			seqBase := kvStart + b*lay.seqLen
			for s := 0; s < lay.seqLen; s++ {
				tok := b*lay.seqLen + s
				kIdx := (seqBase + s) % msl
				for i := 0; i < kvDim; i++ {
					gKT.Data[tok*kvDim+i] = core.FromFloat64[T](gK[kIdx*kvDim+i])
					gVT.Data[tok*kvDim+i] = core.FromFloat64[T](gV[kIdx*kvDim+i])
				}
			}
		}
		gradInK, gradWK, err = dense.Backward(l.K, gKT, flatIn, kPre)
		if err != nil {
			return nil, nil, fmt.Errorf("mha K bwd: %w", err)
		}
		gradInV, gradWV, err = dense.Backward(l.V, gVT, flatIn, vPre)
		if err != nil {
			return nil, nil, fmt.Errorf("mha V bwd: %w", err)
		}
		gradInFlat := core.NewTensor[T](bs, d)
		for i := 0; i < bs*d; i++ {
			sum := core.AsFloat64(gradInQ.Data[i]) + core.AsFloat64(gradInK.Data[i]) + core.AsFloat64(gradInV.Data[i])
			gradInFlat.Data[i] = core.FromFloat64[T](sum)
		}
		gradIn = reshapeSeq(gradInFlat, lay, d)
	}

	gradW = core.NewTensor[T](l.GradWSize())
	off := 0
	off = copyGrad(gradW, off, gradWQ)
	off = copyGrad(gradW, off, gradWK)
	off = copyGrad(gradW, off, gradWV)
	_ = copyGrad(gradW, off, gradWO)
	return gradIn, gradW, nil
}

func copyGrad[T core.Numeric](dst *core.Tensor[T], off int, src *core.Tensor[T]) int {
	n := src.Len()
	copy(dst.Data[off:off+n], src.Data)
	return off + n
}
