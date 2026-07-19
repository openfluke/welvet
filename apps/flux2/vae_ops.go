package flux2

import (
	"fmt"
	"math"
	"runtime"
	"sync"
)

// nchw is a batch-1 NCHW tensor stored as flat [C*H*W] (N omitted).
type nchw struct {
	C, H, W int
	Data    []float32
}

func newNCHW(c, h, w int) nchw {
	return nchw{C: c, H: h, W: w, Data: make([]float32, c*h*w)}
}

func (t nchw) plane(c int) []float32 {
	hw := t.H * t.W
	return t.Data[c*hw : (c+1)*hw]
}

func (t nchw) at(c, y, x int) float32 {
	return t.Data[(c*t.H+y)*t.W+x]
}

func (t nchw) set(c, y, x int, v float32) {
	t.Data[(c*t.H+y)*t.W+x] = v
}

func silu(x float32) float32 {
	// x * sigmoid(x)
	return x / (1 + float32(math.Exp(float64(-x))))
}

func siluInPlace(x []float32) {
	for i, v := range x {
		x[i] = silu(v)
	}
}

// conv2d holds PyTorch-layout Conv2d weights: [Out, In, kH, kW] + optional bias.
type conv2d struct {
	OutC, InC, KH, KW, Pad int
	Weight                 []float32 // Out*In*KH*KW
	Bias                   []float32 // Out (optional)
	Name                   string
}

func (c *conv2d) forward(in nchw) (nchw, error) {
	if c == nil {
		return nchw{}, fmt.Errorf("conv2d: nil")
	}
	if in.C != c.InC {
		return nchw{}, fmt.Errorf("conv2d %s: in C=%d want %d", c.Name, in.C, c.InC)
	}
	outH := in.H + 2*c.Pad - c.KH + 1
	outW := in.W + 2*c.Pad - c.KW + 1
	if outH < 1 || outW < 1 {
		return nchw{}, fmt.Errorf("conv2d %s: bad spatial %dx%d", c.Name, outH, outW)
	}
	out := newNCHW(c.OutC, outH, outW)

	// Fast paths dominate VAE decode time (1×1 shortcuts + 3×3 resnets).
	switch {
	case c.KH == 1 && c.KW == 1 && c.Pad == 0:
		conv2d1x1(c, in, out)
	case c.KH == 3 && c.KW == 3 && c.Pad == 1 && outH == in.H && outW == in.W:
		conv2d3x3pad1(c, in, out)
	default:
		conv2dGeneric(c, in, out)
	}
	return out, nil
}

func convWorkers(outC int) int {
	n := runtime.GOMAXPROCS(0)
	if n < 1 {
		n = 1
	}
	if n > outC {
		n = outC
	}
	if n > 32 {
		n = 32
	}
	return n
}

func parallelOutChannels(outC int, fn func(oc0, oc1 int)) {
	workers := convWorkers(outC)
	if workers <= 1 {
		fn(0, outC)
		return
	}
	var wg sync.WaitGroup
	chunk := (outC + workers - 1) / workers
	for w := 0; w < workers; w++ {
		oc0 := w * chunk
		oc1 := oc0 + chunk
		if oc0 >= outC {
			break
		}
		if oc1 > outC {
			oc1 = outC
		}
		wg.Add(1)
		go func(oc0, oc1 int) {
			defer wg.Done()
			fn(oc0, oc1)
		}(oc0, oc1)
	}
	wg.Wait()
}

// conv2d1x1: out[oc] = bias + Σ_ic w[oc,ic] * in[ic]  (same H×W).
func conv2d1x1(c *conv2d, in, out nchw) {
	hw := in.H * in.W
	parallelOutChannels(c.OutC, func(oc0, oc1 int) {
		for oc := oc0; oc < oc1; oc++ {
			dst := out.Data[oc*hw : (oc+1)*hw]
			var b float32
			if c.Bias != nil {
				b = c.Bias[oc]
			}
			for i := range dst {
				dst[i] = b
			}
			wBase := oc * c.InC
			for ic := 0; ic < c.InC; ic++ {
				w := c.Weight[wBase+ic]
				if w == 0 {
					continue
				}
				src := in.Data[ic*hw : (ic+1)*hw]
				for i, v := range src {
					dst[i] += v * w
				}
			}
		}
	})
}

// conv2d3x3pad1: same-size 3×3; interior pixels skip bounds checks.
func conv2d3x3pad1(c *conv2d, in, out nchw) {
	H, W := in.H, in.W
	inHW := H * W
	outHW := out.H * out.W
	parallelOutChannels(c.OutC, func(oc0, oc1 int) {
		for oc := oc0; oc < oc1; oc++ {
			wBase := oc * c.InC * 9
			dst := out.Data[oc*outHW : (oc+1)*outHW]
			var b float32
			if c.Bias != nil {
				b = c.Bias[oc]
			}
			// Borders (pad=1): safe gather.
			for oy := 0; oy < H; oy++ {
				for ox := 0; ox < W; ox++ {
					if oy > 0 && oy < H-1 && ox > 0 && ox < W-1 {
						continue
					}
					var acc float32
					for ic := 0; ic < c.InC; ic++ {
						srcBase := ic * inHW
						wRow := wBase + ic*9
						for ky := 0; ky < 3; ky++ {
							iy := oy - 1 + ky
							if iy < 0 || iy >= H {
								continue
							}
							row := srcBase + iy*W
							for kx := 0; kx < 3; kx++ {
								ix := ox - 1 + kx
								if ix < 0 || ix >= W {
									continue
								}
								acc += in.Data[row+ix] * c.Weight[wRow+ky*3+kx]
							}
						}
					}
					dst[oy*W+ox] = acc + b
				}
			}
			if H < 3 || W < 3 {
				continue
			}
			// Interior: fully unrolled 3×3, no bounds.
			for oy := 1; oy < H-1; oy++ {
				for ox := 1; ox < W-1; ox++ {
					var acc float32
					for ic := 0; ic < c.InC; ic++ {
						srcBase := ic * inHW
						wRow := wBase + ic*9
						r0 := srcBase + (oy-1)*W + (ox - 1)
						r1 := srcBase + oy*W + (ox - 1)
						r2 := srcBase + (oy+1)*W + (ox - 1)
						ww := c.Weight[wRow : wRow+9]
						acc += in.Data[r0+0]*ww[0] + in.Data[r0+1]*ww[1] + in.Data[r0+2]*ww[2]
						acc += in.Data[r1+0]*ww[3] + in.Data[r1+1]*ww[4] + in.Data[r1+2]*ww[5]
						acc += in.Data[r2+0]*ww[6] + in.Data[r2+1]*ww[7] + in.Data[r2+2]*ww[8]
					}
					dst[oy*W+ox] = acc + b
				}
			}
		}
	})
}

func conv2dGeneric(c *conv2d, in, out nchw) {
	kArea := c.KH * c.KW
	inHW := in.H * in.W
	outH, outW := out.H, out.W
	outHW := outH * outW
	parallelOutChannels(c.OutC, func(oc0, oc1 int) {
		for oc := oc0; oc < oc1; oc++ {
			wBase := oc * c.InC * kArea
			dst := out.Data[oc*outHW : (oc+1)*outHW]
			var b float32
			if c.Bias != nil {
				b = c.Bias[oc]
			}
			for oy := 0; oy < outH; oy++ {
				for ox := 0; ox < outW; ox++ {
					var acc float32
					for ic := 0; ic < c.InC; ic++ {
						srcBase := ic * inHW
						wRow := wBase + ic*kArea
						for ky := 0; ky < c.KH; ky++ {
							iy := oy - c.Pad + ky
							if iy < 0 || iy >= in.H {
								continue
							}
							for kx := 0; kx < c.KW; kx++ {
								ix := ox - c.Pad + kx
								if ix < 0 || ix >= in.W {
									continue
								}
								acc += in.Data[srcBase+iy*in.W+ix] * c.Weight[wRow+ky*c.KW+kx]
							}
						}
					}
					dst[oy*outW+ox] = acc + b
				}
			}
		}
	})
}

// groupNorm is affine GroupNorm over NCHW (N=1): groups along channel.
type groupNorm struct {
	Groups, Channels int
	Eps              float32
	Weight, Bias     []float32 // length Channels
	Name             string
}

func (g *groupNorm) forward(in nchw) (nchw, error) {
	if g == nil {
		return nchw{}, fmt.Errorf("groupNorm: nil")
	}
	if in.C != g.Channels {
		return nchw{}, fmt.Errorf("groupNorm %s: C=%d want %d", g.Name, in.C, g.Channels)
	}
	if g.Channels%g.Groups != 0 {
		return nchw{}, fmt.Errorf("groupNorm %s: channels %% groups", g.Name)
	}
	out := newNCHW(in.C, in.H, in.W)
	chPer := g.Channels / g.Groups
	hw := in.H * in.W
	n := float32(chPer * hw)
	for grp := 0; grp < g.Groups; grp++ {
		c0 := grp * chPer
		var sum, sumSq float32
		for c := 0; c < chPer; c++ {
			plane := in.Data[(c0+c)*hw : (c0+c+1)*hw]
			for _, v := range plane {
				sum += v
				sumSq += v * v
			}
		}
		mean := sum / n
		var_ := sumSq/n - mean*mean
		if var_ < 0 {
			var_ = 0
		}
		inv := 1 / float32(math.Sqrt(float64(var_+g.Eps)))
		for c := 0; c < chPer; c++ {
			ci := c0 + c
			src := in.Data[ci*hw : (ci+1)*hw]
			dst := out.Data[ci*hw : (ci+1)*hw]
			w, b := g.Weight[ci], g.Bias[ci]
			for i, v := range src {
				dst[i] = (v-mean)*inv*w + b
			}
		}
	}
	return out, nil
}

// nearestUpsample2x doubles H and W (PyTorch F.interpolate nearest).
func nearestUpsample2x(in nchw) nchw {
	out := newNCHW(in.C, in.H*2, in.W*2)
	for c := 0; c < in.C; c++ {
		for y := 0; y < in.H; y++ {
			for x := 0; x < in.W; x++ {
				v := in.at(c, y, x)
				out.set(c, y*2, x*2, v)
				out.set(c, y*2, x*2+1, v)
				out.set(c, y*2+1, x*2, v)
				out.set(c, y*2+1, x*2+1, v)
			}
		}
	}
	return out
}

// resnetBlock2D is Diffusers ResnetBlock2D without time embedding (VAE path).
type resnetBlock2D struct {
	Norm1, Norm2   *groupNorm
	Conv1, Conv2   *conv2d
	ConvShortcut   *conv2d // optional 1x1
	OutputScale    float32
	Name           string
}

func (r *resnetBlock2D) forward(in nchw) (nchw, error) {
	h, err := r.Norm1.forward(in)
	if err != nil {
		return nchw{}, err
	}
	siluInPlace(h.Data)
	h, err = r.Conv1.forward(h)
	if err != nil {
		return nchw{}, err
	}
	h, err = r.Norm2.forward(h)
	if err != nil {
		return nchw{}, err
	}
	siluInPlace(h.Data)
	h, err = r.Conv2.forward(h)
	if err != nil {
		return nchw{}, err
	}
	skip := in
	if r.ConvShortcut != nil {
		skip, err = r.ConvShortcut.forward(in)
		if err != nil {
			return nchw{}, err
		}
	}
	scale := r.OutputScale
	if scale == 0 {
		scale = 1
	}
	out := newNCHW(h.C, h.H, h.W)
	for i := range out.Data {
		out.Data[i] = (skip.Data[i] + h.Data[i]) / scale
	}
	return out, nil
}

// vaeAttention is Diffusers Attention used in UNetMidBlock2D (self-attn, residual).
// Gap note: uses classic softmax attention (AttnProcessor); AttnProcessor2_0 SDPA
// is numerically equivalent for this single-head mid-block case.
type vaeAttention struct {
	Channels           int
	Heads              int
	DimHead            int
	Scale              float32
	RescaleOutputFactor float32
	GroupNorm          *groupNorm
	ToQ, ToK, ToV, ToOut *Linear
	Name               string
}

func (a *vaeAttention) forward(in nchw) (nchw, error) {
	if a == nil {
		return in, nil
	}
	residual := in
	// GroupNorm over channels with spatial flattened: [1,C,HW]
	gnIn := nchw{C: in.C, H: in.H * in.W, W: 1, Data: in.Data}
	normed, err := a.GroupNorm.forward(gnIn)
	if err != nil {
		return nchw{}, err
	}
	seq := in.H * in.W
	// tokens [seq, C] from CHW
	tok := make([]float32, seq*a.Channels)
	for c := 0; c < a.Channels; c++ {
		for s := 0; s < seq; s++ {
			tok[s*a.Channels+c] = normed.Data[c*seq+s]
		}
	}
	q := make([]float32, seq*a.Channels)
	k := make([]float32, seq*a.Channels)
	v := make([]float32, seq*a.Channels)
	if err := a.ToQ.MatMulSeq(tok, q, seq); err != nil {
		return nchw{}, err
	}
	if err := a.ToK.MatMulSeq(tok, k, seq); err != nil {
		return nchw{}, err
	}
	if err := a.ToV.MatMulSeq(tok, v, seq); err != nil {
		return nchw{}, err
	}

	// Single- or multi-head attention. heads * dim_head == channels.
	heads := a.Heads
	dh := a.DimHead
	outTok := make([]float32, seq*a.Channels)
	for h := 0; h < heads; h++ {
		// scores [seq, seq]
		scores := make([]float32, seq*seq)
		for i := 0; i < seq; i++ {
			qi := q[i*a.Channels+h*dh : i*a.Channels+h*dh+dh]
			rowMax := float32(-math.MaxFloat32)
			for j := 0; j < seq; j++ {
				kj := k[j*a.Channels+h*dh : j*a.Channels+h*dh+dh]
				var dot float32
				for t := 0; t < dh; t++ {
					dot += qi[t] * kj[t]
				}
				s := dot * a.Scale
				scores[i*seq+j] = s
				if s > rowMax {
					rowMax = s
				}
			}
			var sum float32
			for j := 0; j < seq; j++ {
				e := float32(math.Exp(float64(scores[i*seq+j] - rowMax)))
				scores[i*seq+j] = e
				sum += e
			}
			inv := 1 / sum
			for j := 0; j < seq; j++ {
				scores[i*seq+j] *= inv
			}
		}
		for i := 0; i < seq; i++ {
			for t := 0; t < dh; t++ {
				var acc float32
				for j := 0; j < seq; j++ {
					acc += scores[i*seq+j] * v[j*a.Channels+h*dh+t]
				}
				outTok[i*a.Channels+h*dh+t] = acc
			}
		}
	}
	proj := make([]float32, seq*a.Channels)
	if err := a.ToOut.MatMulSeq(outTok, proj, seq); err != nil {
		return nchw{}, err
	}
	out := newNCHW(in.C, in.H, in.W)
	for c := 0; c < a.Channels; c++ {
		for s := 0; s < seq; s++ {
			out.Data[c*seq+s] = proj[s*a.Channels+c]
		}
	}
	scale := a.RescaleOutputFactor
	if scale == 0 {
		scale = 1
	}
	for i := range out.Data {
		out.Data[i] = (out.Data[i] + residual.Data[i]) / scale
	}
	return out, nil
}

type vaeMidBlock struct {
	Res0, Res1 *resnetBlock2D
	Attn       *vaeAttention
}

func (m *vaeMidBlock) forward(in nchw) (nchw, error) {
	h, err := m.Res0.forward(in)
	if err != nil {
		return nchw{}, err
	}
	if m.Attn != nil {
		h, err = m.Attn.forward(h)
		if err != nil {
			return nchw{}, err
		}
	}
	return m.Res1.forward(h)
}

type vaeUpBlock struct {
	Resnets   []*resnetBlock2D
	Upsampler *conv2d // after nearest 2x; nil on final block
}

func (u *vaeUpBlock) forward(in nchw) (nchw, error) {
	h := in
	var err error
	for _, r := range u.Resnets {
		h, err = r.forward(h)
		if err != nil {
			return nchw{}, err
		}
	}
	if u.Upsampler != nil {
		h = nearestUpsample2x(h)
		h, err = u.Upsampler.forward(h)
		if err != nil {
			return nchw{}, err
		}
	}
	return h, nil
}
