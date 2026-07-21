package mha

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/webgpu"
)

// ForwardWebGPU — real device required (no host fake). Q/K/V/O projections run
// on-device via dense.ForwardWebGPU. When gpuAttnSupported, RoPE (if PosRoPE),
// QK-RMSNorm (if enabled), and tiled softmax attention also run on device.
// Unsupported configs fall back to the full forwardHost path.
//
// BackwardWebGPU: when gpuAttnSupported, attention backward ALU runs on device
// (webgpu.MHABackward); RoPE transpose stays host; projections on device.
func ForwardWebGPU[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("mha: BackendWebGPU but no device (no host fake)")
	}
	if !gpuAttnSupported(l) {
		return forwardHost(l, input)
	}
	return forwardWebGPUAttn(l, input)
}

// BackwardWebGPU — reverse of ForwardWebGPU.
func BackwardWebGPU[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("mha: BackendWebGPU but no device (no host fake)")
	}
	if !gpuAttnSupported(l) {
		return backwardHost(l, gradOut, input, pre)
	}
	return backwardWebGPUAttn(l, gradOut, input, pre)
}

func forwardWebGPUAttn[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	cfg := l.Cfg
	lay, err := parseLayout(cfg.DModel, input)
	if err != nil {
		return nil, nil, err
	}
	qDim, kvDim := cfg.QDim(), cfg.KVDim()
	msl := cfg.MaxSeqLen
	d := cfg.DModel
	tileSize := l.Exec.TileSize
	if tileSize <= 0 {
		tileSize = l.Core.TileSize
	}
	if tileSize <= 0 {
		tileSize = 32
	}

	flatIn := flattenTokens(input, lay)
	_, qPost, err := dense.Forward(l.Q, flatIn)
	if err != nil {
		return nil, nil, fmt.Errorf("mha Q: %w", err)
	}
	_, kPost, err := dense.Forward(l.K, flatIn)
	if err != nil {
		return nil, nil, fmt.Errorf("mha K: %w", err)
	}
	_, vPost, err := dense.Forward(l.V, flatIn)
	if err != nil {
		return nil, nil, fmt.Errorf("mha V: %w", err)
	}
	prepareKV(l, lay, msl, kvDim)
	kvStart := l.KVOffset

	attnFlat := core.NewTensor[T](lay.batch*lay.seqLen, qDim)
	qGammaF := f64ToF32(l.QNormWeight)
	kGammaF := f64ToF32(l.KNormWeight)
	theta := float32(cfg.RoPETheta)
	causal := cfg.Mask == MaskCausal

	needScratch := lay.seqLen * qDim
	if needScratch > len(l.DecodeScratchAttn) {
		l.DecodeScratchAttn = make([]float64, needScratch)
	}

	for b := 0; b < lay.batch; b++ {
		seqBase := kvStart + b*lay.seqLen
		qF := make([]float32, lay.seqLen*qDim)
		kF := make([]float32, lay.seqLen*kvDim)
		vF := make([]float32, lay.seqLen*kvDim)
		for s := 0; s < lay.seqLen; s++ {
			tok := b*lay.seqLen + s
			for i := 0; i < qDim; i++ {
				qF[s*qDim+i] = float32(core.AsFloat64(qPost.Data[tok*qDim+i]))
			}
			for i := 0; i < kvDim; i++ {
				kF[s*kvDim+i] = float32(core.AsFloat64(kPost.Data[tok*kvDim+i]))
				vF[s*kvDim+i] = float32(core.AsFloat64(vPost.Data[tok*kvDim+i]))
			}
		}

		if cfg.QKNorm {
			eps := float32(cfg.QKNormEps)
			if err := webgpu.RMSNorm(qF, qGammaF, qF, lay.seqLen*cfg.NumHeads, cfg.HeadDim, eps); err != nil {
				return nil, nil, fmt.Errorf("mha Q norm GPU: %w", err)
			}
			if err := webgpu.RMSNorm(kF, kGammaF, kF, lay.seqLen*cfg.NumKVHeads, cfg.HeadDim, eps); err != nil {
				return nil, nil, fmt.Errorf("mha K norm GPU: %w", err)
			}
		}

		if cfg.UsesRoPE() {
			pos := []int32{int32(seqBase)}
			if err := webgpu.RoPEApply(qF, lay.seqLen, cfg.NumHeads, cfg.HeadDim, theta, pos); err != nil {
				return nil, nil, fmt.Errorf("mha RoPE Q GPU: %w", err)
			}
			if err := webgpu.RoPEApply(kF, lay.seqLen, cfg.NumKVHeads, cfg.HeadDim, theta, pos); err != nil {
				return nil, nil, fmt.Errorf("mha RoPE K GPU: %w", err)
			}
		}

		for s := 0; s < lay.seqLen; s++ {
			pos := seqBase + s
			kRow := l.KVCacheK[(pos%msl)*kvDim : (pos%msl+1)*kvDim]
			vRow := l.KVCacheV[(pos%msl)*kvDim : (pos%msl+1)*kvDim]
			copy(kRow, f32ToF64(kF[s*kvDim:(s+1)*kvDim]))
			copy(vRow, f32ToF64(vF[s*kvDim:(s+1)*kvDim]))
		}

		kvLen := seqBase + lay.seqLen
		kCacheGPU := webgpu.PackKVCache(l.KVCacheK, cfg.NumKVHeads, msl, cfg.HeadDim, kvLen)
		vCacheGPU := webgpu.PackKVCache(l.KVCacheV, cfg.NumKVHeads, msl, cfg.HeadDim, kvLen)

		attnF := make([]float32, lay.seqLen*qDim)
		mhaCfg := webgpu.MHAConfig{
			NumHeads: cfg.NumHeads, NumKVHeads: cfg.NumKVHeads, HeadDim: cfg.HeadDim,
			SeqLen: lay.seqLen, KVOffset: seqBase, MaxSeqLen: msl, KvLen: kvLen,
			TileSize: tileSize, Causal: causal,
		}
		if err := webgpu.MHAForward(qF, kCacheGPU, vCacheGPU, attnF, mhaCfg); err != nil {
			return nil, nil, fmt.Errorf("mha attn GPU: %w", err)
		}

		for s := 0; s < lay.seqLen; s++ {
			tok := b*lay.seqLen + s
			for i := 0; i < qDim; i++ {
				attnFlat.Data[tok*qDim+i] = core.FromFloat64[T](float64(attnF[s*qDim+i]))
			}
		}
	}
	l.KVOffset = kvStart + lay.batch*lay.seqLen

	_, oPost, err := dense.Forward(l.O, attnFlat)
	if err != nil {
		return nil, nil, fmt.Errorf("mha O: %w", err)
	}
	pre = reshapeSeq(attnFlat, lay, qDim)
	post = reshapeSeq(oPost, lay, d)
	return pre, post, nil
}

func backwardWebGPUAttn[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	cfg := l.Cfg
	lay, err := parseLayout(cfg.DModel, input)
	if err != nil {
		return nil, nil, err
	}
	qDim, kvDim := cfg.QDim(), cfg.KVDim()
	msl := cfg.MaxSeqLen
	d := cfg.DModel
	bs := lay.batch * lay.seqLen
	causal := cfg.Mask == MaskCausal
	tileSize := l.Exec.TileSize
	if tileSize <= 0 {
		tileSize = 32
	}

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
	kPre, kPost, err := dense.Forward(l.K, flatIn)
	if err != nil {
		return nil, nil, fmt.Errorf("mha K recompute: %w", err)
	}
	vPre, vPost, err := dense.Forward(l.V, flatIn)
	if err != nil {
		return nil, nil, fmt.Errorf("mha V recompute: %w", err)
	}

	kvEnd := l.KVOffset
	if kvEnd < bs {
		kvEnd = bs
	}
	kvStart := kvEnd - bs

	gQAll := make([]float64, bs*qDim)
	gK := make([]float64, msl*kvDim)
	gV := make([]float64, msl*kvDim)
	qGammaF := f64ToF32(l.QNormWeight)
	kGammaF := f64ToF32(l.KNormWeight)
	theta := float32(cfg.RoPETheta)

	for b := 0; b < lay.batch; b++ {
		seqBase := kvStart + b*lay.seqLen
		qF := make([]float32, lay.seqLen*qDim)
		kF := make([]float32, lay.seqLen*kvDim)
		vF := make([]float32, lay.seqLen*kvDim)
		for s := 0; s < lay.seqLen; s++ {
			tok := b*lay.seqLen + s
			for i := 0; i < qDim; i++ {
				qF[s*qDim+i] = float32(core.AsFloat64(qPost.Data[tok*qDim+i]))
			}
			for i := 0; i < kvDim; i++ {
				kF[s*kvDim+i] = float32(core.AsFloat64(kPost.Data[tok*kvDim+i]))
				vF[s*kvDim+i] = float32(core.AsFloat64(vPost.Data[tok*kvDim+i]))
			}
		}
		if cfg.QKNorm {
			eps := float32(cfg.QKNormEps)
			if err := webgpu.RMSNorm(qF, qGammaF, qF, lay.seqLen*cfg.NumHeads, cfg.HeadDim, eps); err != nil {
				return nil, nil, err
			}
			if err := webgpu.RMSNorm(kF, kGammaF, kF, lay.seqLen*cfg.NumKVHeads, cfg.HeadDim, eps); err != nil {
				return nil, nil, err
			}
		}
		if cfg.UsesRoPE() {
			pos := []int32{int32(seqBase)}
			if err := webgpu.RoPEApply(qF, lay.seqLen, cfg.NumHeads, cfg.HeadDim, theta, pos); err != nil {
				return nil, nil, err
			}
			if err := webgpu.RoPEApply(kF, lay.seqLen, cfg.NumKVHeads, cfg.HeadDim, theta, pos); err != nil {
				return nil, nil, err
			}
		}
		// Rebuild host ring for PackKVCache (matches forward).
		cacheK := make([]float64, msl*kvDim)
		cacheV := make([]float64, msl*kvDim)
		for s := 0; s < lay.seqLen; s++ {
			pos := seqBase + s
			copy(cacheK[(pos%msl)*kvDim:(pos%msl+1)*kvDim], f32ToF64(kF[s*kvDim:(s+1)*kvDim]))
			copy(cacheV[(pos%msl)*kvDim:(pos%msl+1)*kvDim], f32ToF64(vF[s*kvDim:(s+1)*kvDim]))
		}
		kvLen := seqBase + lay.seqLen
		kCacheGPU := webgpu.PackKVCache(cacheK, cfg.NumKVHeads, msl, cfg.HeadDim, kvLen)
		vCacheGPU := webgpu.PackKVCache(cacheV, cfg.NumKVHeads, msl, cfg.HeadDim, kvLen)

		gradF := make([]float32, lay.seqLen*qDim)
		for s := 0; s < lay.seqLen; s++ {
			tok := b*lay.seqLen + s
			for i := 0; i < qDim; i++ {
				gradF[s*qDim+i] = float32(core.AsFloat64(gradAttn.Data[tok*qDim+i]))
			}
		}
		dQ := make([]float32, lay.seqLen*qDim)
		dKGPU := make([]float32, cfg.NumKVHeads*msl*cfg.HeadDim)
		dVGPU := make([]float32, cfg.NumKVHeads*msl*cfg.HeadDim)
		mhaCfg := webgpu.MHAConfig{
			NumHeads: cfg.NumHeads, NumKVHeads: cfg.NumKVHeads, HeadDim: cfg.HeadDim,
			SeqLen: lay.seqLen, KVOffset: seqBase, MaxSeqLen: msl, KvLen: kvLen,
			TileSize: tileSize, Causal: causal,
		}
		if err := webgpu.MHABackward(gradF, qF, kCacheGPU, vCacheGPU, dQ, dKGPU, dVGPU, mhaCfg); err != nil {
			return nil, nil, fmt.Errorf("mha attn bwd GPU: %w", err)
		}

		gQ := f32ToF64(dQ)
		unpackKVGrad(dKGPU, gK, cfg.NumKVHeads, msl, cfg.HeadDim, kvLen)
		unpackKVGrad(dVGPU, gV, cfg.NumKVHeads, msl, cfg.HeadDim, kvLen)

		for s := 0; s < lay.seqLen; s++ {
			pos := seqBase + s
			if cfg.UsesRoPE() {
				applyRoPEBackward(gQ[s*qDim:(s+1)*qDim], pos, cfg.NumHeads, cfg.HeadDim, cfg.RoPETheta)
			}
			tok := b*lay.seqLen + s
			copy(gQAll[tok*qDim:(tok+1)*qDim], gQ[s*qDim:(s+1)*qDim])
		}
		if cfg.UsesRoPE() {
			for s := 0; s < lay.seqLen; s++ {
				pos := seqBase + s
				kIdx := pos % msl
				applyRoPEBackward(gK[kIdx*kvDim:(kIdx+1)*kvDim], pos, cfg.NumKVHeads, cfg.HeadDim, cfg.RoPETheta)
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
	gradInK, gradWK, err := dense.Backward(l.K, gKT, flatIn, kPre)
	if err != nil {
		return nil, nil, fmt.Errorf("mha K bwd: %w", err)
	}
	gradInV, gradWV, err := dense.Backward(l.V, gVT, flatIn, vPre)
	if err != nil {
		return nil, nil, fmt.Errorf("mha V bwd: %w", err)
	}
	gradInFlat := core.NewTensor[T](bs, d)
	for i := 0; i < bs*d; i++ {
		sum := core.AsFloat64(gradInQ.Data[i]) + core.AsFloat64(gradInK.Data[i]) + core.AsFloat64(gradInV.Data[i])
		gradInFlat.Data[i] = core.FromFloat64[T](sum)
	}
	gradIn = reshapeSeq(gradInFlat, lay, d)

	gradW = core.NewTensor[T](l.GradWSize())
	off := 0
	off = copyGrad(gradW, off, gradWQ)
	off = copyGrad(gradW, off, gradWK)
	off = copyGrad(gradW, off, gradWV)
	_ = copyGrad(gradW, off, gradWO)
	return gradIn, gradW, nil
}

// unpackKVGrad converts GPU [numKVHeads, maxSeqLen, headDim] grads into host [maxSeqLen, kvDim].
func unpackKVGrad(gpu []float32, host []float64, numKVHeads, maxSeqLen, headDim, kvLen int) {
	kvDim := numKVHeads * headDim
	if kvLen > maxSeqLen {
		kvLen = maxSeqLen
	}
	for kPos := 0; kPos < kvLen; kPos++ {
		dst := host[(kPos%maxSeqLen)*kvDim : (kPos%maxSeqLen+1)*kvDim]
		for h := 0; h < numKVHeads; h++ {
			for d := 0; d < headDim; d++ {
				dst[h*headDim+d] += float64(gpu[(h*maxSeqLen+kPos)*headDim+d])
			}
		}
	}
}

func f64ToF32(in []float64) []float32 {
	if len(in) == 0 {
		return nil
	}
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(v)
	}
	return out
}

func f32ToF64(in []float32) []float64 {
	out := make([]float64, len(in))
	for i, v := range in {
		out[i] = float64(v)
	}
	return out
}
