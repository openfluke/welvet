package qwentts

import (
	"fmt"
	"math"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/openfluke/welvet/model/hf"
	"github.com/openfluke/welvet/simd"
)

// convWeight is a 1-D conv kernel [Cout, Cin, K] row-major + optional bias.
type convWeight struct {
	w          []float32
	b          []float32
	cout, cin  int
	k          int
}

type snakeAct struct{ alpha, beta []float32 }

type convnextBlk struct {
	dwW, dwB    []float32 // depthwise [C,1,7]
	normW, normB []float32
	pw1, pw2    *Linear
	gamma       []float32
}

type upsampleBlk struct {
	trans  convWeight // transposed conv [Cin,Cout,K]
	stride int
	cnext  convnextBlk
}

type residUnit struct {
	act1   snakeAct
	conv1  convWeight
	act2   snakeAct
	conv2  convWeight
}

type decoderBlk struct {
	snake  snakeAct
	trans  convWeight
	stride int
	res    [3]residUnit
}

type decLayer struct {
	inNorm, postNorm    []float32
	q, k, v, o          *Linear
	attnScale, mlpScale []float32
	gate, up, down      *Linear
}

// Decoder is the Qwen3-TTS speech tokenizer decoder (SplitRVQ + pre_transformer
// + ConvNeXt upsample + SnakeBeta conv decoder) producing 24 kHz mono PCM.
type Decoder struct {
	cfg DecoderConfig

	// quantizer
	cbFirst [][]float32 // 1 codebook  [2048*256]
	cbRest  [][]float32 // 15 codebooks
	cbDim   int         // 256
	outProjFirst convWeight
	outProjRest  convWeight

	preConv convWeight

	inputProj  *Linear
	preLayers  []decLayer
	preNorm    []float32
	outputProj *Linear
	rope       *ropeCache

	upsample []upsampleBlk

	initConv convWeight
	blocks   []decoderBlk
	finSnake snakeAct
	finConv  convWeight

	numQuant int
}

func loadConv(path string, idx map[string]hf.TensorInfo, name string, bias bool) (convWeight, error) {
	ti, ok := idx[name+".weight"]
	if !ok {
		return convWeight{}, fmt.Errorf("missing %s.weight", name)
	}
	if len(ti.Shape) != 3 {
		return convWeight{}, fmt.Errorf("%s.weight: want 3-D conv, got %v", name, ti.Shape)
	}
	w, err := hf.LoadF16Vector(path, idx, name+".weight")
	if err != nil {
		return convWeight{}, err
	}
	// Conv1d weight layout: [out_channels, in_channels, kernel]
	cw := convWeight{w: w, cout: ti.Shape[0], cin: ti.Shape[1], k: ti.Shape[2]}
	if bias {
		if cw.b, err = hf.LoadF16Vector(path, idx, name+".bias"); err != nil {
			return convWeight{}, err
		}
	}
	return cw, nil
}

// loadConvTranspose loads ConvTranspose1d weights: [in_channels, out_channels, kernel].
func loadConvTranspose(path string, idx map[string]hf.TensorInfo, name string, bias bool) (convWeight, error) {
	ti, ok := idx[name+".weight"]
	if !ok {
		return convWeight{}, fmt.Errorf("missing %s.weight", name)
	}
	if len(ti.Shape) != 3 {
		return convWeight{}, fmt.Errorf("%s.weight: want 3-D convtranspose, got %v", name, ti.Shape)
	}
	w, err := hf.LoadF16Vector(path, idx, name+".weight")
	if err != nil {
		return convWeight{}, err
	}
	cw := convWeight{w: w, cin: ti.Shape[0], cout: ti.Shape[1], k: ti.Shape[2]}
	if bias {
		if cw.b, err = hf.LoadF16Vector(path, idx, name+".bias"); err != nil {
			return convWeight{}, err
		}
	}
	return cw, nil
}

func loadSnake(path string, idx map[string]hf.TensorInfo, name string) (snakeAct, error) {
	a, err := hf.LoadF16Vector(path, idx, name+".alpha")
	if err != nil {
		return snakeAct{}, err
	}
	b, err := hf.LoadF16Vector(path, idx, name+".beta")
	if err != nil {
		return snakeAct{}, err
	}
	return snakeAct{alpha: a, beta: b}, nil
}

func loadDecoder(snapshotDir string, cfg DecoderConfig) (*Decoder, error) {
	path := filepath.Join(snapshotDir, "speech_tokenizer", "model.safetensors")
	idx, err := hf.BuildTensorIndex(path)
	if err != nil {
		return nil, fmt.Errorf("decoder index: %w", err)
	}
	d := &Decoder{cfg: cfg, numQuant: cfg.NumQuantizers}

	// quantizer codebooks (EMA reconstruction)
	buildCB := func(prefix string, n int) ([][]float32, error) {
		out := make([][]float32, n)
		for i := 0; i < n; i++ {
			cu, err := hf.LoadF16Vector(path, idx, fmt.Sprintf("%s.vq.layers.%d._codebook.cluster_usage", prefix, i))
			if err != nil {
				return nil, err
			}
			es, err := hf.LoadF16Vector(path, idx, fmt.Sprintf("%s.vq.layers.%d._codebook.embedding_sum", prefix, i))
			if err != nil {
				return nil, err
			}
			dim := len(es) / len(cu)
			d.cbDim = dim
			emb := make([]float32, len(es))
			for r := 0; r < len(cu); r++ {
				den := cu[r]
				if den < 1e-5 {
					den = 1e-5
				}
				for c := 0; c < dim; c++ {
					emb[r*dim+c] = es[r*dim+c] / den
				}
			}
			out[i] = emb
		}
		return out, nil
	}
	if d.cbFirst, err = buildCB("decoder.quantizer.rvq_first", 1); err != nil {
		return nil, err
	}
	nRest := cfg.NumQuantizers - 1
	if d.cbRest, err = buildCB("decoder.quantizer.rvq_rest", nRest); err != nil {
		return nil, err
	}
	if d.outProjFirst, err = loadConv(path, idx, "decoder.quantizer.rvq_first.output_proj", false); err != nil {
		return nil, err
	}
	if d.outProjRest, err = loadConv(path, idx, "decoder.quantizer.rvq_rest.output_proj", false); err != nil {
		return nil, err
	}

	if d.preConv, err = loadConv(path, idx, "decoder.pre_conv.conv", true); err != nil {
		return nil, err
	}

	// pre_transformer
	if d.inputProj, err = loadLinear(path, idx, "decoder.pre_transformer.input_proj", true); err != nil {
		return nil, err
	}
	if d.outputProj, err = loadLinear(path, idx, "decoder.pre_transformer.output_proj", true); err != nil {
		return nil, err
	}
	if d.preNorm, err = loadVec(path, idx, "decoder.pre_transformer.norm.weight"); err != nil {
		return nil, err
	}
	d.preLayers = make([]decLayer, cfg.NumLayers)
	for i := 0; i < cfg.NumLayers; i++ {
		lp := fmt.Sprintf("decoder.pre_transformer.layers.%d", i)
		var l decLayer
		if l.inNorm, err = loadVec(path, idx, lp+".input_layernorm.weight"); err != nil {
			return nil, err
		}
		if l.postNorm, err = loadVec(path, idx, lp+".post_attention_layernorm.weight"); err != nil {
			return nil, err
		}
		if l.q, err = loadLinear(path, idx, lp+".self_attn.q_proj", false); err != nil {
			return nil, err
		}
		if l.k, err = loadLinear(path, idx, lp+".self_attn.k_proj", false); err != nil {
			return nil, err
		}
		if l.v, err = loadLinear(path, idx, lp+".self_attn.v_proj", false); err != nil {
			return nil, err
		}
		if l.o, err = loadLinear(path, idx, lp+".self_attn.o_proj", false); err != nil {
			return nil, err
		}
		if l.attnScale, err = loadVec(path, idx, lp+".self_attn_layer_scale.scale"); err != nil {
			return nil, err
		}
		if l.mlpScale, err = loadVec(path, idx, lp+".mlp_layer_scale.scale"); err != nil {
			return nil, err
		}
		if l.gate, err = loadLinear(path, idx, lp+".mlp.gate_proj", false); err != nil {
			return nil, err
		}
		if l.up, err = loadLinear(path, idx, lp+".mlp.up_proj", false); err != nil {
			return nil, err
		}
		if l.down, err = loadLinear(path, idx, lp+".mlp.down_proj", false); err != nil {
			return nil, err
		}
		d.preLayers[i] = l
	}
	d.rope = newRopeCache(cfg.HeadDim, 8192, cfg.RopeTheta)

	// upsample (ConvNeXt) blocks
	d.upsample = make([]upsampleBlk, len(cfg.UpsamplingRatios))
	for i, ratio := range cfg.UpsamplingRatios {
		up := upsampleBlk{stride: ratio}
		bp := fmt.Sprintf("decoder.upsample.%d", i)
		if up.trans, err = loadConvTranspose(path, idx, bp+".0.conv", true); err != nil {
			return nil, err
		}
		cn := convnextBlk{}
		if cw, e := loadConv(path, idx, bp+".1.dwconv.conv", true); e == nil {
			cn.dwW, cn.dwB = cw.w, cw.b
		} else {
			return nil, e
		}
		if cn.normW, err = loadVec(path, idx, bp+".1.norm.weight"); err != nil {
			return nil, err
		}
		if cn.normB, err = loadVec(path, idx, bp+".1.norm.bias"); err != nil {
			return nil, err
		}
		if cn.pw1, err = loadLinear(path, idx, bp+".1.pwconv1", true); err != nil {
			return nil, err
		}
		if cn.pw2, err = loadLinear(path, idx, bp+".1.pwconv2", true); err != nil {
			return nil, err
		}
		if cn.gamma, err = loadVec(path, idx, bp+".1.gamma"); err != nil {
			return nil, err
		}
		up.cnext = cn
		d.upsample[i] = up
	}

	// conv decoder
	if d.initConv, err = loadConv(path, idx, "decoder.decoder.0.conv", true); err != nil {
		return nil, err
	}
	d.blocks = make([]decoderBlk, len(cfg.UpsampleRates))
	for i, rate := range cfg.UpsampleRates {
		bp := fmt.Sprintf("decoder.decoder.%d", i+1)
		blk := decoderBlk{stride: rate}
		if blk.snake, err = loadSnake(path, idx, bp+".block.0"); err != nil {
			return nil, err
		}
		if blk.trans, err = loadConvTranspose(path, idx, bp+".block.1.conv", true); err != nil {
			return nil, err
		}
		for j := 0; j < 3; j++ {
			rp := fmt.Sprintf("%s.block.%d", bp, j+2)
			var ru residUnit
			if ru.act1, err = loadSnake(path, idx, rp+".act1"); err != nil {
				return nil, err
			}
			if ru.conv1, err = loadConv(path, idx, rp+".conv1.conv", true); err != nil {
				return nil, err
			}
			if ru.act2, err = loadSnake(path, idx, rp+".act2"); err != nil {
				return nil, err
			}
			if ru.conv2, err = loadConv(path, idx, rp+".conv2.conv", true); err != nil {
				return nil, err
			}
			blk.res[j] = ru
		}
		d.blocks[i] = blk
	}
	nb := len(cfg.UpsampleRates)
	if d.finSnake, err = loadSnake(path, idx, fmt.Sprintf("decoder.decoder.%d", nb+1)); err != nil {
		return nil, err
	}
	if d.finConv, err = loadConv(path, idx, fmt.Sprintf("decoder.decoder.%d.conv", nb+2), true); err != nil {
		return nil, err
	}
	return d, nil
}

// ---- conv/activation primitives (channel-first [C*T]) ----

// axpyShifted: dst[t] += alpha * src[t-shift] for valid t in [0,T).
func axpyShifted(dst, src []float32, alpha float32, shift, T int) {
	if alpha == 0 || T <= 0 {
		return
	}
	if shift >= 0 {
		n := T - shift
		if n <= 0 {
			return
		}
		simd.SaxpyF32(dst[shift:shift+n], alpha, src[:n], n)
		return
	}
	s := -shift
	n := T - s
	if n <= 0 {
		return
	}
	simd.SaxpyF32(dst[:n], alpha, src[s:s+n], n)
}

func causalConv(in []float32, cin, T int, cw convWeight, dilation int) []float32 {
	cout := cw.cout
	k := cw.k
	padLeft := (k - 1) * dilation
	out := make([]float32, cout*T)
	if cw.b != nil {
		for oc := 0; oc < cout; oc++ {
			simd.FillF32(out[oc*T:oc*T+T], cw.b[oc], T)
		}
	}
	// Pointwise (k=1): SIMD saxpy over time for each (oc,ic).
	if k == 1 && dilation == 1 {
		workers := runtime.GOMAXPROCS(0)
		if workers < 1 {
			workers = 1
		}
		if cout < 8 || T < 256 || workers == 1 {
			for oc := 0; oc < cout; oc++ {
				causalConvPointwiseOC(in, out, cin, T, oc, cw.w)
			}
			return out
		}
		var wg sync.WaitGroup
		ch := make(chan int, cout)
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for oc := range ch {
					causalConvPointwiseOC(in, out, cin, T, oc, cw.w)
				}
			}()
		}
		for oc := 0; oc < cout; oc++ {
			ch <- oc
		}
		close(ch)
		wg.Wait()
		return out
	}
	// Dilated causal: SIMD saxpy on each (ic,j) time-shift.
	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}
	if cout < workers {
		workers = cout
	}
	if workers == 1 || T < 256 {
		for oc := 0; oc < cout; oc++ {
			causalConvOutChan(in, out, cin, T, oc, cw.w, k, padLeft, dilation)
		}
		return out
	}
	var wg sync.WaitGroup
	ch := make(chan int, cout)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for oc := range ch {
				causalConvOutChan(in, out, cin, T, oc, cw.w, k, padLeft, dilation)
			}
		}()
	}
	for oc := 0; oc < cout; oc++ {
		ch <- oc
	}
	close(ch)
	wg.Wait()
	return out
}

func causalConvPointwiseOC(in, out []float32, cin, T, oc int, w []float32) {
	wbase := oc * cin
	orow := out[oc*T : oc*T+T]
	for ic := 0; ic < cin; ic++ {
		wj := w[wbase+ic]
		if wj == 0 {
			continue
		}
		simd.SaxpyF32(orow, wj, in[ic*T:ic*T+T], T)
	}
}

func causalConvOutChan(in, out []float32, cin, T, oc int, w []float32, k, padLeft, dilation int) {
	wbase := oc * cin * k
	orow := out[oc*T : oc*T+T]
	for ic := 0; ic < cin; ic++ {
		irow := in[ic*T : ic*T+T]
		wrow := w[wbase+ic*k : wbase+ic*k+k]
		for j := 0; j < k; j++ {
			wj := wrow[j]
			if wj == 0 {
				continue
			}
			// tin = t - padLeft + j*dilation  =>  shift = padLeft - j*dilation
			axpyShifted(orow, irow, wj, padLeft-j*dilation, T)
		}
	}
}

func depthwiseCausalConv(in []float32, c, T int, w, b []float32, k int) []float32 {
	padLeft := k - 1
	out := make([]float32, c*T)
	for ch := 0; ch < c; ch++ {
		orow := out[ch*T : ch*T+T]
		if b != nil {
			simd.FillF32(orow, b[ch], T)
		}
		wrow := w[ch*k : ch*k+k]
		irow := in[ch*T : ch*T+T]
		for j := 0; j < k; j++ {
			wj := wrow[j]
			if wj == 0 {
				continue
			}
			axpyShifted(orow, irow, wj, padLeft-j, T)
		}
	}
	return out
}

// transposedConv upsamples time by stride. weight [Cin,Cout,K]. Returns [Cout*Tout].
func transposedConv(in []float32, cin, T int, cw convWeight, stride int) ([]float32, int) {
	if cw.cin > 0 {
		cin = cw.cin
	}
	cout := cw.cout
	k := cw.k
	need := cin * cout * k
	if cin <= 0 || cout <= 0 || k <= 0 || T <= 0 || len(cw.w) < need || len(in) < cin*T {
		return nil, 0
	}
	rightPad := k - stride
	if rightPad < 0 {
		rightPad = 0
	}
	toutFull := (T-1)*stride + k
	tmp := make([]float32, cout*toutFull)
	for ic := 0; ic < cin; ic++ {
		irow := in[ic*T : ic*T+T]
		wbase := ic * cout * k
		for oc := 0; oc < cout; oc++ {
			off := wbase + oc*k
			wrow := cw.w[off : off+k]
			obase := oc * toutFull
			for j := 0; j < k; j++ {
				wj := wrow[j]
				if wj == 0 {
					continue
				}
				// tmp[oc, t*stride+j] += wj * in[ic,t]
				simd.AddScaledF32Stride(tmp, obase+j, stride, wj, irow, T)
			}
		}
	}
	tout := toutFull - rightPad
	if tout < 1 {
		tout = 1
	}
	out := make([]float32, cout*tout)
	for oc := 0; oc < cout; oc++ {
		var bias float32
		if cw.b != nil {
			bias = cw.b[oc]
		}
		src := tmp[oc*toutFull : oc*toutFull+tout]
		dst := out[oc*tout : oc*tout+tout]
		if bias == 0 {
			copy(dst, src)
			continue
		}
		simd.FillF32(dst, bias, tout)
		simd.SaxpyF32(dst, 1, src, tout)
	}
	return out, tout
}

func applySnake(x []float32, c, T int, s snakeAct) {
	for ch := 0; ch < c; ch++ {
		a := float32(math.Exp(float64(s.alpha[ch])))
		b := float32(math.Exp(float64(s.beta[ch])))
		inv := 1.0 / (b + 1e-9)
		row := x[ch*T : ch*T+T]
		for t := 0; t < T; t++ {
			sn := float32(math.Sin(float64(a * row[t])))
			row[t] = row[t] + inv*sn*sn
		}
	}
}

func (d *Decoder) convnext(x []float32, c, T int, cn convnextBlk) []float32 {
	y := depthwiseCausalConv(x, c, T, cn.dwW, cn.dwB, 7)
	// per-timestep: layernorm(C) -> pw1 -> gelu -> pw2 -> gamma
	col := make([]float32, c)
	h1 := make([]float32, cn.pw1.Out)
	h2 := make([]float32, cn.pw2.Out)
	out := make([]float32, c*T)
	for t := 0; t < T; t++ {
		for ch := 0; ch < c; ch++ {
			col[ch] = y[ch*T+t]
		}
		layerNorm(col, cn.normW, cn.normB, c, 1e-6)
		cn.pw1.forward(col, h1)
		for i := range h1 {
			h1[i] = gelu(h1[i])
		}
		cn.pw2.forward(h1, h2)
		for ch := 0; ch < c; ch++ {
			out[ch*T+t] = x[ch*T+t] + cn.gamma[ch]*h2[ch]
		}
	}
	return out
}

// preTransformer runs the 8-layer sliding-window transformer on [T*latent] ->
// returns [T*latent] (channel-last, row-major [T][latent]).
func (d *Decoder) preTransformer(x []float32, T int) []float32 {
	cfg := d.cfg
	hidden := d.inputProj.Out // 512
	hd := cfg.HeadDim
	qDim := d.preLayers[0].q.Out
	kvDim := d.preLayers[0].k.Out
	nh := qDim / hd
	nkv := kvDim / hd
	rep := nh / nkv
	win := cfg.SlidingWindow
	eps := cfg.RMSNormEps
	scale := float32(1 / math.Sqrt(float64(hd)))

	// input_proj: [T*latent] -> [T*hidden]
	h := make([]float32, T*hidden)
	d.inputProj.forwardSeq(x, h, T)

	xn := make([]float32, T*hidden)
	qAll := make([]float32, T*qDim)
	kAll := make([]float32, T*kvDim)
	vAll := make([]float32, T*kvDim)
	attn := make([]float32, T*qDim)
	tmp := make([]float32, T*hidden)
	gate := make([]float32, d.preLayers[0].gate.Out)
	up := make([]float32, d.preLayers[0].gate.Out)
	scores := make([]float32, T)

	for li := range d.preLayers {
		l := &d.preLayers[li]
		for t := 0; t < T; t++ {
			rmsNormTo(xn[t*hidden:(t+1)*hidden], h[t*hidden:(t+1)*hidden], l.inNorm, hidden, eps)
		}
		l.q.forwardSeq(xn, qAll, T)
		l.k.forwardSeq(xn, kAll, T)
		l.v.forwardSeq(xn, vAll, T)
		for t := 0; t < T; t++ {
			for hh := 0; hh < nh; hh++ {
				d.rope.applyRope(qAll[t*qDim+hh*hd:t*qDim+hh*hd+hd], t)
			}
			for hh := 0; hh < nkv; hh++ {
				d.rope.applyRope(kAll[t*kvDim+hh*hd:t*kvDim+hh*hd+hd], t)
			}
		}
		for t := 0; t < T; t++ {
			start := 0
			if win > 0 && t-win+1 > 0 {
				start = t - win + 1
			}
			for hh := 0; hh < nh; hh++ {
				kvh := hh / rep
				qh := qAll[t*qDim+hh*hd : t*qDim+hh*hd+hd]
				n := 0
				for tp := start; tp <= t; tp++ {
					kh := kAll[tp*kvDim+kvh*hd : tp*kvDim+kvh*hd+hd]
					scores[n] = dotF32(qh, kh, hd) * scale
					n++
				}
				softmaxInPlace(scores, n)
				out := attn[t*qDim+hh*hd : t*qDim+hh*hd+hd]
				for dd := 0; dd < hd; dd++ {
					out[dd] = 0
				}
				n = 0
				for tp := start; tp <= t; tp++ {
					w := scores[n]
					vh := vAll[tp*kvDim+kvh*hd : tp*kvDim+kvh*hd+hd]
					for dd := 0; dd < hd; dd++ {
						out[dd] += w * vh[dd]
					}
					n++
				}
			}
		}
		l.o.forwardSeq(attn, tmp, T)
		for t := 0; t < T; t++ {
			for dd := 0; dd < hidden; dd++ {
				h[t*hidden+dd] += l.attnScale[dd] * tmp[t*hidden+dd]
			}
		}
		for t := 0; t < T; t++ {
			rmsNormTo(xn[t*hidden:(t+1)*hidden], h[t*hidden:(t+1)*hidden], l.postNorm, hidden, eps)
			l.gate.forward(xn[t*hidden:(t+1)*hidden], gate)
			l.up.forward(xn[t*hidden:(t+1)*hidden], up)
			siluMul(gate, up)
			l.down.forward(gate, tmp[t*hidden:(t+1)*hidden])
		}
		for t := 0; t < T; t++ {
			for dd := 0; dd < hidden; dd++ {
				h[t*hidden+dd] += l.mlpScale[dd] * tmp[t*hidden+dd]
			}
		}
	}
	for t := 0; t < T; t++ {
		rmsNorm(h[t*hidden:(t+1)*hidden], d.preNorm, hidden, eps)
	}
	// output_proj: [T*hidden] -> [T*latent]
	lat := d.outputProj.Out
	out := make([]float32, T*lat)
	d.outputProj.forwardSeq(h, out, T)
	return out
}

// rvqDecode sums codebook lookups over the given codebooks then applies
// output_proj (Conv1d k=1) -> [512*T] channel-first.
func (d *Decoder) rvqDecode(codes [][]int, cbs [][]float32, outProj convWeight, T int) []float32 {
	dim := d.cbDim
	acc := make([]float32, dim*T) // [dim][T]
	for l := 0; l < len(cbs); l++ {
		cb := cbs[l]
		row := codes[l]
		for t := 0; t < T; t++ {
			code := row[t]
			if code < 0 {
				code = 0
			}
			emb := cb[code*dim : code*dim+dim]
			for c := 0; c < dim; c++ {
				acc[c*T+t] += emb[c]
			}
		}
	}
	return causalConv(acc, dim, T, outProj, 1) // k=1 conv == pointwise proj
}

// decode turns [T][16] codec codes into 24 kHz mono PCM.
func (d *Decoder) decode(codes [][]int) ([]float32, error) {
	T := len(codes)
	if T == 0 {
		return nil, fmt.Errorf("qwentts decode: no frames")
	}
	fmt.Printf("  qwen decode: %d frames → ~%d samples @ %d Hz…\n", T, T*1920, d.cfg.SampleRate)
	nq := d.numQuant
	// transpose to [q][T]
	perQ := make([][]int, nq)
	for q := 0; q < nq; q++ {
		perQ[q] = make([]int, T)
		for t := 0; t < T; t++ {
			if q < len(codes[t]) {
				perQ[q][t] = codes[t][q]
			}
		}
	}
	// SplitRVQ decode
	fmt.Println("  qwen decode: RVQ + pre-conv…")
	q0 := d.rvqDecode(perQ[:1], d.cbFirst, d.outProjFirst, T) // [512*T]
	qr := d.rvqDecode(perQ[1:nq], d.cbRest, d.outProjRest, T) // [512*T]
	cbdim := d.cfg.CodebookDim
	quant := make([]float32, cbdim*T)
	for i := range quant {
		quant[i] = q0[i] + qr[i]
	}
	// pre_conv -> [latent*T]
	lat := d.cfg.LatentDim
	h := causalConv(quant, cbdim, T, d.preConv, 1) // [latent*T]
	// -> channel-last [T*latent]
	hcl := make([]float32, T*lat)
	for c := 0; c < lat; c++ {
		for t := 0; t < T; t++ {
			hcl[t*lat+c] = h[c*T+t]
		}
	}
	// pre_transformer -> [T*latent]
	fmt.Println("  qwen decode: pre-transformer…")
	hcl = d.preTransformer(hcl, T)
	// -> channel-first [latent*T]
	cur := make([]float32, lat*T)
	for t := 0; t < T; t++ {
		for c := 0; c < lat; c++ {
			cur[c*T+t] = hcl[t*lat+c]
		}
	}
	curC := lat
	curT := T
	// upsample ConvNeXt blocks
	for i := range d.upsample {
		up := &d.upsample[i]
		fmt.Printf("  qwen decode: upsample[%d] stride=%d (T=%d)…\n", i, up.stride, curT)
		var nt int
		cur, nt = transposedConv(cur, curC, curT, up.trans, up.stride)
		if cur == nil || nt == 0 {
			return nil, fmt.Errorf("qwentts decode: upsample[%d] transposedConv failed (cin=%d T=%d)", i, curC, curT)
		}
		curC = up.trans.cout
		curT = nt
		cur = d.convnext(cur, curC, curT, up.cnext)
	}
	// initial conv -> decoder_dim
	fmt.Printf("  qwen decode: snake decoder (T=%d)…\n", curT)
	cur = causalConv(cur, curC, curT, d.initConv, 1)
	curC = d.initConv.cout
	// decoder blocks
	for i := range d.blocks {
		blk := &d.blocks[i]
		fmt.Printf("  qwen decode: block[%d] stride=%d (T=%d → %d)…\n", i, blk.stride, curT, curT*blk.stride)
		applySnake(cur, curC, curT, blk.snake)
		var nt int
		cur, nt = transposedConv(cur, curC, curT, blk.trans, blk.stride)
		if cur == nil || nt == 0 {
			return nil, fmt.Errorf("qwentts decode: decoder block[%d] transposedConv failed (cin=%d T=%d)", i, curC, curT)
		}
		curC = blk.trans.cout
		curT = nt
		for j := 0; j < 3; j++ {
			ru := &blk.res[j]
			dil := []int{1, 3, 9}[j]
			res := make([]float32, len(cur))
			copy(res, cur)
			applySnake(cur, curC, curT, ru.act1)
			cur = causalConv(cur, curC, curT, ru.conv1, dil)
			applySnake(cur, curC, curT, ru.act2)
			cur = causalConv(cur, curC, curT, ru.conv2, 1)
			for x := range cur {
				cur[x] += res[x]
			}
		}
	}
	applySnake(cur, curC, curT, d.finSnake)
	cur = causalConv(cur, curC, curT, d.finConv, 1)
	// mono + clamp
	out := make([]float32, curT)
	for t := 0; t < curT; t++ {
		v := cur[t]
		if v > 1 {
			v = 1
		} else if v < -1 {
			v = -1
		}
		out[t] = v
	}
	fmt.Printf("  qwen decode: done (%d samples)\n", len(out))
	return out, nil
}

// SetFuse enables SIMD / sticky GPU GEMV on decoder transformer + ConvNeXt linears.
func (d *Decoder) SetFuse(simdOn, gpuOn bool) {
	if d == nil {
		return
	}
	setLinearFuseMany(simdOn, gpuOn, d.inputProj, d.outputProj)
	for i := range d.preLayers {
		l := &d.preLayers[i]
		setLinearFuseMany(simdOn, gpuOn, l.q, l.k, l.v, l.o, l.gate, l.up, l.down)
	}
	for i := range d.upsample {
		setLinearFuseMany(simdOn, gpuOn, d.upsample[i].cnext.pw1, d.upsample[i].cnext.pw2)
	}
}

// WarmGPU uploads large decoder linears.
// Returns early with IsF32VRAMFull when the sticky soft-cap is hit.
func (d *Decoder) WarmGPU() (int, error) {
	if d == nil {
		return 0, nil
	}
	n := 0
	warm := func(l *Linear) error {
		if l == nil || !l.UseGPU {
			return nil
		}
		if err := warmLinearGPU(l); err != nil {
			return err
		}
		n++
		return nil
	}
	for _, l := range []*Linear{d.inputProj, d.outputProj} {
		if err := warm(l); err != nil {
			return n, err
		}
	}
	for i := range d.preLayers {
		l := &d.preLayers[i]
		for _, lin := range []*Linear{l.q, l.k, l.v, l.o, l.gate, l.up, l.down} {
			if err := warm(lin); err != nil {
				return n, err
			}
		}
	}
	for i := range d.upsample {
		if err := warm(d.upsample[i].cnext.pw1); err != nil {
			return n, err
		}
		if err := warm(d.upsample[i].cnext.pw2); err != nil {
			return n, err
		}
	}
	return n, nil
}
