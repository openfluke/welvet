package flux2

import (
	"fmt"
	"time"
	"unsafe"

	"github.com/openfluke/webgpu/wgpu"
	"github.com/openfluke/welvet/webgpu"
)

// vaeGPU holds resident decoder weights + scratch for fused WebGPU decode.
type vaeGPU struct {
	dev   *wgpu.Device
	queue *wgpu.Queue

	pipeSilu, pipeAdd, pipeCopy, pipeUp      *wgpu.ComputePipeline
	pipeConv, pipeGNStats, pipeGNApply       *wgpu.ComputePipeline
	pipeGEMV, pipeAttn, pipeToTok, pipeToCHW *wgpu.ComputePipeline

	wBuf  map[uintptr]*wgpu.Buffer
	owned []*wgpu.Buffer

	scratch  [4]*wgpu.Buffer
	attnQKV  [3]*wgpu.Buffer
	stats    *wgpu.Buffer
	maxElems int
	maxGroups int
	attnElems int
	bytes    uint64
	ready    bool
}

func (v *AutoencoderKLFlux2) SyncGPU(maxPixelSide int) error {
	if v == nil || !v.Loaded {
		return fmt.Errorf("VAE.SyncGPU: not loaded")
	}
	if !webgpu.Available() {
		if err := webgpu.InitError(); err != nil {
			return err
		}
		return fmt.Errorf("webgpu: no adapter")
	}
	if maxPixelSide <= 0 {
		maxPixelSide = 512
	}
	peakC := 512
	for _, c := range v.BlockOutChannels {
		if c > peakC {
			peakC = c
		}
	}
	// Nearest upsample keeps channels before the upsampler conv, so peak is
	// full-res × max(block_out) — e.g. 256²×512 after up_blocks.1→2 upsample.
	maxElems := maxPixelSide * maxPixelSide * peakC
	if maxElems < 1<<20 {
		maxElems = 1 << 20
	}

	v.CloseGPU()
	g := &vaeGPU{wBuf: make(map[uintptr]*wgpu.Buffer), maxElems: maxElems, maxGroups: 64}
	err := webgpu.WithDevice(func(dev *wgpu.Device, q *wgpu.Queue) error {
		g.dev, g.queue = dev, q
		return g.build(v)
	})
	if err != nil {
		_ = webgpu.WithDevice(func(dev *wgpu.Device, q *wgpu.Queue) error {
			g.destroyLocked()
			return nil
		})
		return err
	}
	v.gpu = g
	fmt.Printf("  flux2 VAE GPU: resident decoder (%.0f MiB, scratch %d elems for ≤%dpx)\n",
		float64(g.bytes)/(1024*1024), g.maxElems, maxPixelSide)
	return nil
}

func (v *AutoencoderKLFlux2) CloseGPU() {
	if v == nil || v.gpu == nil {
		return
	}
	g := v.gpu
	v.gpu = nil
	_ = webgpu.WithDevice(func(dev *wgpu.Device, q *wgpu.Queue) error {
		dev.Poll(true, nil)
		g.destroyLocked()
		return nil
	})
}

func (g *vaeGPU) destroyLocked() {
	for _, b := range g.owned {
		if b != nil {
			b.Destroy()
		}
	}
	g.owned = nil
	g.wBuf = nil
	g.ready = false
}

func (g *vaeGPU) own(b *wgpu.Buffer, n uint64) {
	g.owned = append(g.owned, b)
	g.bytes += n
}

func (g *vaeGPU) mkEmpty(label string, elems int) (*wgpu.Buffer, error) {
	size := uint64(elems * 4)
	if size < 64 {
		size = 64
	}
	b, err := g.dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: label, Size: size,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, err
	}
	g.own(b, size)
	return b, nil
}

func (g *vaeGPU) uploadF32(label string, data []float32) (*wgpu.Buffer, error) {
	if len(data) == 0 {
		data = []float32{0}
	}
	b, err := g.dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: label, Contents: wgpu.ToBytes(data),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, err
	}
	g.own(b, uint64(len(data)*4))
	return b, nil
}

func (g *vaeGPU) ensureW(key uintptr, label string, data []float32) (*wgpu.Buffer, error) {
	if b, ok := g.wBuf[key]; ok {
		return b, nil
	}
	b, err := g.uploadF32(label, data)
	if err != nil {
		return nil, err
	}
	g.wBuf[key] = b
	return b, nil
}

func (g *vaeGPU) convWB(c *conv2d) (w, b *wgpu.Buffer, has uint32, err error) {
	w, err = g.ensureW(uintptr(unsafe.Pointer(&c.Weight[0])), c.Name+".w", c.Weight)
	if err != nil {
		return
	}
	if c.Bias != nil && len(c.Bias) >= c.OutC {
		b, err = g.ensureW(uintptr(unsafe.Pointer(&c.Bias[0])), c.Name+".b", c.Bias)
		has = 1
		return
	}
	b, err = g.ensureW(uintptr(unsafe.Pointer(c))^1, c.Name+".bz", make([]float32, c.OutC))
	return
}

func (g *vaeGPU) gnWB(gn *groupNorm) (gamma, beta *wgpu.Buffer, err error) {
	gamma, err = g.ensureW(uintptr(unsafe.Pointer(&gn.Weight[0])), gn.Name+".g", gn.Weight)
	if err != nil {
		return
	}
	beta, err = g.ensureW(uintptr(unsafe.Pointer(&gn.Bias[0])), gn.Name+".b", gn.Bias)
	return
}

func (g *vaeGPU) linWB(l *Linear) (w, b *wgpu.Buffer, has uint32, err error) {
	w, err = g.ensureW(uintptr(unsafe.Pointer(&l.Weight[0])), l.Name+".w", l.Weight)
	if err != nil {
		return
	}
	if l.Bias != nil && len(l.Bias) >= l.Out {
		b, err = g.ensureW(uintptr(unsafe.Pointer(&l.Bias[0])), l.Name+".b", l.Bias)
		has = 1
		return
	}
	b, err = g.ensureW(uintptr(unsafe.Pointer(l))^1, l.Name+".bz", make([]float32, l.Out))
	return
}

func (g *vaeGPU) build(v *AutoencoderKLFlux2) error {
	var err error
	mk := func(code, label string) *wgpu.ComputePipeline {
		if err != nil {
			return nil
		}
		var p *wgpu.ComputePipeline
		p, err = webgpu.MakeComputePipeline(g.dev, code, label)
		return p
	}
	g.pipeSilu = mk(shaderVAESilu, "vae-silu")
	g.pipeAdd = mk(shaderVAEAdd, "vae-add")
	g.pipeCopy = mk(shaderVAECopy, "vae-copy")
	g.pipeUp = mk(shaderVAEUpsample2x, "vae-up")
	g.pipeConv = mk(shaderVAEConv, "vae-conv")
	g.pipeGNStats = mk(shaderVAEGNStats, "vae-gns")
	g.pipeGNApply = mk(shaderVAEGNApply, "vae-gna")
	g.pipeGEMV = mk(shaderVAEGEMV, "vae-gemv")
	g.pipeAttn = mk(shaderVAEAttn, "vae-attn")
	g.pipeToTok = mk(shaderVAENCHWToTok, "vae-tok")
	g.pipeToCHW = mk(shaderVAETokToNCHW, "vae-chw")
	if err != nil {
		return err
	}

	touchC := func(c *conv2d) {
		if err != nil || c == nil {
			return
		}
		_, _, _, err = g.convWB(c)
	}
	touchG := func(gn *groupNorm) {
		if err != nil || gn == nil {
			return
		}
		_, _, err = g.gnWB(gn)
	}
	touchL := func(l *Linear) {
		if err != nil || l == nil {
			return
		}
		_, _, _, err = g.linWB(l)
	}
	touchC(v.PostQuant)
	touchC(v.ConvIn)
	touchG(v.NormOut)
	touchC(v.ConvOut)
	if v.Mid != nil {
		for _, r := range []*resnetBlock2D{v.Mid.Res0, v.Mid.Res1} {
			if r == nil {
				continue
			}
			touchG(r.Norm1)
			touchG(r.Norm2)
			touchC(r.Conv1)
			touchC(r.Conv2)
			touchC(r.ConvShortcut)
		}
		if a := v.Mid.Attn; a != nil {
			touchG(a.GroupNorm)
			touchL(a.ToQ)
			touchL(a.ToK)
			touchL(a.ToV)
			touchL(a.ToOut)
			g.attnElems = 64 * 64 * a.Channels // enough for 512px (lat 64)
			for i := 0; i < 3; i++ {
				g.attnQKV[i], err = g.mkEmpty(fmt.Sprintf("vae-qkv-%d", i), g.attnElems)
				if err != nil {
					return err
				}
			}
		}
	}
	for _, up := range v.UpBlocks {
		if up == nil {
			continue
		}
		for _, r := range up.Resnets {
			if r == nil {
				continue
			}
			touchG(r.Norm1)
			touchG(r.Norm2)
			touchC(r.Conv1)
			touchC(r.Conv2)
			touchC(r.ConvShortcut)
		}
		touchC(up.Upsampler)
	}
	if err != nil {
		return err
	}
	for i := 0; i < 4; i++ {
		g.scratch[i], err = g.mkEmpty(fmt.Sprintf("vae-s-%d", i), g.maxElems)
		if err != nil {
			return err
		}
	}
	g.stats, err = g.mkEmpty("vae-stats", g.maxGroups*2)
	if err != nil {
		return err
	}
	g.ready = true
	return nil
}

type gTensor struct {
	C, H, W, Buf int
}

func (g *vaeGPU) n(t gTensor) int { return t.C * t.H * t.W }

func (g *vaeGPU) sb(i int) *wgpu.Buffer {
	if i >= 0 && i < 4 {
		return g.scratch[i]
	}
	return g.scratch[0]
}

func wgCeil(n, ws int) uint32 {
	if n < 1 {
		return 1
	}
	return uint32((n + ws - 1) / ws)
}

// wgGrid2D splits workgroups so no dimension exceeds WebGPU's 65535 limit.
// Shaders must use: let i = gid.y * (65535u * workgroupSize) + gid.x;
func wgGrid2D(n, ws int) (x, y uint32) {
	total := wgCeil(n, ws)
	const maxDim = 65535
	if total <= maxDim {
		return total, 1
	}
	return maxDim, (total + maxDim - 1) / maxDim
}

func (g *vaeGPU) uni(data []byte) (*wgpu.Buffer, error) {
	return g.dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "vae-u", Contents: data,
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
}

func (g *vaeGPU) disp(pipe *wgpu.ComputePipeline, e []wgpu.BindGroupEntry, x, y, z uint32) error {
	bg, err := g.dev.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: pipe.GetBindGroupLayout(0), Entries: e})
	if err != nil {
		return err
	}
	defer bg.Release()
	enc, err := g.dev.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipe)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(x, y, z)
	pass.End()
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	g.queue.Submit(cmd)
	return nil
}

// dispFlat dispatches a 1D kernel over n elements with workgroup size ws,
// splitting into a 2D grid when n/ws > 65535 (WebGPU limit).
func (g *vaeGPU) dispFlat(pipe *wgpu.ComputePipeline, e []wgpu.BindGroupEntry, n, ws int) error {
	x, y := wgGrid2D(n, ws)
	return g.disp(pipe, e, x, y, 1)
}

func (g *vaeGPU) free(used ...int) int {
	u := map[int]bool{}
	for _, x := range used {
		u[x] = true
	}
	for i := 0; i < 4; i++ {
		if !u[i] {
			return i
		}
	}
	return 0
}

func (g *vaeGPU) silu(t gTensor) error {
	type p struct{ N, A, B, C uint32 }
	u, err := g.uni(wgpu.ToBytes([]p{{N: uint32(g.n(t))}}))
	if err != nil {
		return err
	}
	defer u.Destroy()
	return g.dispFlat(g.pipeSilu, []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: u, Size: u.GetSize()},
		{Binding: 1, Buffer: g.sb(t.Buf), Size: g.sb(t.Buf).GetSize()},
	}, g.n(t), 256)
}

func (g *vaeGPU) add(a, b, out gTensor, scale float32) error {
	if scale == 0 {
		scale = 1
	}
	type p struct {
		N      uint32
		Scale  float32
		P1, P2 uint32
	}
	u, err := g.uni(wgpu.ToBytes([]p{{N: uint32(g.n(out)), Scale: scale}}))
	if err != nil {
		return err
	}
	defer u.Destroy()
	return g.dispFlat(g.pipeAdd, []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: u, Size: u.GetSize()},
		{Binding: 1, Buffer: g.sb(a.Buf), Size: g.sb(a.Buf).GetSize()},
		{Binding: 2, Buffer: g.sb(b.Buf), Size: g.sb(b.Buf).GetSize()},
		{Binding: 3, Buffer: g.sb(out.Buf), Size: g.sb(out.Buf).GetSize()},
	}, g.n(out), 256)
}

func (g *vaeGPU) upsample(in gTensor, outBuf int) (gTensor, error) {
	out := gTensor{C: in.C, H: in.H * 2, W: in.W * 2, Buf: outBuf}
	if g.n(out) > g.maxElems {
		return out, fmt.Errorf("VAE GPU scratch too small for %dx%dx%d — SyncGPU larger side", out.C, out.H, out.W)
	}
	type p struct{ C, H, W, P uint32 }
	u, err := g.uni(wgpu.ToBytes([]p{{C: uint32(in.C), H: uint32(in.H), W: uint32(in.W)}}))
	if err != nil {
		return out, err
	}
	defer u.Destroy()
	err = g.dispFlat(g.pipeUp, []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: u, Size: u.GetSize()},
		{Binding: 1, Buffer: g.sb(in.Buf), Size: g.sb(in.Buf).GetSize()},
		{Binding: 2, Buffer: g.sb(out.Buf), Size: g.sb(out.Buf).GetSize()},
	}, g.n(out), 256)
	return out, err
}

func (g *vaeGPU) conv(c *conv2d, in gTensor, outBuf int) (gTensor, error) {
	out := gTensor{C: c.OutC, H: in.H + 2*c.Pad - c.KH + 1, W: in.W + 2*c.Pad - c.KW + 1, Buf: outBuf}
	if g.n(out) > g.maxElems {
		return out, fmt.Errorf("VAE GPU conv %s scratch", c.Name)
	}
	w, b, has, err := g.convWB(c)
	if err != nil {
		return out, err
	}
	type p struct{ OutC, InC, H, W, KH, KW, Pad, Has uint32 }
	u, err := g.uni(wgpu.ToBytes([]p{{
		OutC: uint32(c.OutC), InC: uint32(c.InC), H: uint32(in.H), W: uint32(in.W),
		KH: uint32(c.KH), KW: uint32(c.KW), Pad: uint32(c.Pad), Has: has,
	}}))
	if err != nil {
		return out, err
	}
	defer u.Destroy()
	err = g.dispFlat(g.pipeConv, []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: u, Size: u.GetSize()},
		{Binding: 1, Buffer: g.sb(in.Buf), Size: g.sb(in.Buf).GetSize()},
		{Binding: 2, Buffer: w, Size: w.GetSize()},
		{Binding: 3, Buffer: b, Size: b.GetSize()},
		{Binding: 4, Buffer: g.sb(out.Buf), Size: g.sb(out.Buf).GetSize()},
	}, g.n(out), 256)
	return out, err
}

func (g *vaeGPU) groupNorm(gn *groupNorm, in gTensor, outBuf int) (gTensor, error) {
	out := gTensor{C: in.C, H: in.H, W: in.W, Buf: outBuf}
	gamma, beta, err := g.gnWB(gn)
	if err != nil {
		return out, err
	}
	type ps struct {
		Ch, Gr, H, W uint32
		Eps          float32
		A, B, C      uint32
	}
	u1, err := g.uni(wgpu.ToBytes([]ps{{Ch: uint32(gn.Channels), Gr: uint32(gn.Groups), H: uint32(in.H), W: uint32(in.W), Eps: gn.Eps}}))
	if err != nil {
		return out, err
	}
	defer u1.Destroy()
	if err := g.disp(g.pipeGNStats, []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: u1, Size: u1.GetSize()},
		{Binding: 1, Buffer: g.sb(in.Buf), Size: g.sb(in.Buf).GetSize()},
		{Binding: 2, Buffer: g.stats, Size: g.stats.GetSize()},
	}, uint32(gn.Groups), 1, 1); err != nil {
		return out, err
	}
	type pa struct{ Ch, Gr, H, W, A, B, C, D uint32 }
	u2, err := g.uni(wgpu.ToBytes([]pa{{Ch: uint32(gn.Channels), Gr: uint32(gn.Groups), H: uint32(in.H), W: uint32(in.W)}}))
	if err != nil {
		return out, err
	}
	defer u2.Destroy()
	err = g.dispFlat(g.pipeGNApply, []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: u2, Size: u2.GetSize()},
		{Binding: 1, Buffer: g.sb(in.Buf), Size: g.sb(in.Buf).GetSize()},
		{Binding: 2, Buffer: g.stats, Size: g.stats.GetSize()},
		{Binding: 3, Buffer: gamma, Size: gamma.GetSize()},
		{Binding: 4, Buffer: beta, Size: beta.GetSize()},
		{Binding: 5, Buffer: g.sb(out.Buf), Size: g.sb(out.Buf).GetSize()},
	}, g.n(out), 256)
	return out, err
}

func (g *vaeGPU) gemv(l *Linear, x, y *wgpu.Buffer, seq int) error {
	w, b, has, err := g.linWB(l)
	if err != nil {
		return err
	}
	type p struct{ Batch, In, Out, Has uint32 }
	u, err := g.uni(wgpu.ToBytes([]p{{Batch: uint32(seq), In: uint32(l.In), Out: uint32(l.Out), Has: has}}))
	if err != nil {
		return err
	}
	defer u.Destroy()
	return g.disp(g.pipeGEMV, []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: u, Size: u.GetSize()},
		{Binding: 1, Buffer: x, Size: x.GetSize()},
		{Binding: 2, Buffer: w, Size: w.GetSize()},
		{Binding: 3, Buffer: b, Size: b.GetSize()},
		{Binding: 4, Buffer: y, Size: y.GetSize()},
	}, wgCeil(l.Out, 64), uint32(seq), 1)
}

func (g *vaeGPU) attnSDPA(q, k, v, out *wgpu.Buffer, seq, dim int, scale float32) error {
	type p struct {
		Seq, Dim uint32
		Scale    float32
		P        uint32
	}
	u, err := g.uni(wgpu.ToBytes([]p{{Seq: uint32(seq), Dim: uint32(dim), Scale: scale}}))
	if err != nil {
		return err
	}
	defer u.Destroy()
	return g.disp(g.pipeAttn, []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: u, Size: u.GetSize()},
		{Binding: 1, Buffer: q, Size: q.GetSize()},
		{Binding: 2, Buffer: k, Size: k.GetSize()},
		{Binding: 3, Buffer: v, Size: v.GetSize()},
		{Binding: 4, Buffer: out, Size: out.GetSize()},
	}, wgCeil(seq, 64), 1, 1)
}

func (g *vaeGPU) toTok(in gTensor, outBuf int) error {
	type p struct{ C, H, W, P uint32 }
	u, err := g.uni(wgpu.ToBytes([]p{{C: uint32(in.C), H: uint32(in.H), W: uint32(in.W)}}))
	if err != nil {
		return err
	}
	defer u.Destroy()
	return g.dispFlat(g.pipeToTok, []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: u, Size: u.GetSize()},
		{Binding: 1, Buffer: g.sb(in.Buf), Size: g.sb(in.Buf).GetSize()},
		{Binding: 2, Buffer: g.sb(outBuf), Size: g.sb(outBuf).GetSize()},
	}, g.n(in), 256)
}

func (g *vaeGPU) toCHW(c, h, w, inBuf, outBuf int) error {
	type p struct{ C, H, W, P uint32 }
	u, err := g.uni(wgpu.ToBytes([]p{{C: uint32(c), H: uint32(h), W: uint32(w)}}))
	if err != nil {
		return err
	}
	defer u.Destroy()
	return g.dispFlat(g.pipeToCHW, []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: u, Size: u.GetSize()},
		{Binding: 1, Buffer: g.sb(inBuf), Size: g.sb(inBuf).GetSize()},
		{Binding: 2, Buffer: g.sb(outBuf), Size: g.sb(outBuf).GetSize()},
	}, c*h*w, 256)
}

func (g *vaeGPU) resnet(r *resnetBlock2D, in gTensor) (gTensor, error) {
	t1 := g.free(in.Buf)
	t2 := g.free(in.Buf, t1)
	h, err := g.groupNorm(r.Norm1, in, t1)
	if err != nil {
		return in, err
	}
	if err := g.silu(h); err != nil {
		return in, err
	}
	h, err = g.conv(r.Conv1, h, t2)
	if err != nil {
		return in, err
	}
	h, err = g.groupNorm(r.Norm2, h, t1)
	if err != nil {
		return in, err
	}
	if err := g.silu(h); err != nil {
		return in, err
	}
	h, err = g.conv(r.Conv2, h, t2)
	if err != nil {
		return in, err
	}
	skip := in
	if r.ConvShortcut != nil {
		skip, err = g.conv(r.ConvShortcut, in, t1)
		if err != nil {
			return in, err
		}
	}
	outBuf := g.free(h.Buf, skip.Buf)
	out := gTensor{C: h.C, H: h.H, W: h.W, Buf: outBuf}
	scale := r.OutputScale
	if scale == 0 {
		scale = 1
	}
	if err := g.add(skip, h, out, scale); err != nil {
		return in, err
	}
	return out, nil
}

func (g *vaeGPU) attention(a *vaeAttention, in gTensor) (gTensor, error) {
	if a == nil {
		return in, nil
	}
	seq := in.H * in.W
	need := seq * a.Channels
	if need > g.attnElems {
		return in, fmt.Errorf("VAE GPU attn: seq %d too large for scratch", seq)
	}
	t1 := g.free(in.Buf)
	t2 := g.free(in.Buf, t1)
	t3 := g.free(in.Buf, t1, t2)

	gnIn := gTensor{C: in.C, H: seq, W: 1, Buf: in.Buf}
	normed, err := g.groupNorm(a.GroupNorm, gnIn, t1)
	if err != nil {
		return in, err
	}
	normed.H, normed.W = in.H, in.W
	if err := g.toTok(normed, t2); err != nil {
		return in, err
	}
	tok := g.sb(t2)
	if err := g.gemv(a.ToQ, tok, g.attnQKV[0], seq); err != nil {
		return in, err
	}
	if err := g.gemv(a.ToK, tok, g.attnQKV[1], seq); err != nil {
		return in, err
	}
	if err := g.gemv(a.ToV, tok, g.attnQKV[2], seq); err != nil {
		return in, err
	}
	// attn out tokens → t1
	if err := g.attnSDPA(g.attnQKV[0], g.attnQKV[1], g.attnQKV[2], g.sb(t1), seq, a.Channels, a.Scale); err != nil {
		return in, err
	}
	if err := g.gemv(a.ToOut, g.sb(t1), g.sb(t2), seq); err != nil {
		return in, err
	}
	if err := g.toCHW(in.C, in.H, in.W, t2, t3); err != nil {
		return in, err
	}
	out := gTensor{C: in.C, H: in.H, W: in.W, Buf: t1}
	scale := a.RescaleOutputFactor
	if scale == 0 {
		scale = 1
	}
	if err := g.add(gTensor{C: in.C, H: in.H, W: in.W, Buf: t3}, in, out, scale); err != nil {
		return in, err
	}
	return out, nil
}

func (g *vaeGPU) decode(v *AutoencoderKLFlux2, z []float32, latH, latW int) ([]float32, error) {
	t0 := time.Now()
	c := v.LatentChannels
	need := c * latH * latW
	if len(z) < need {
		return nil, fmt.Errorf("VAE GPU decode: short latent")
	}
	if need > g.maxElems {
		return nil, fmt.Errorf("VAE GPU: latent %d > scratch %d", need, g.maxElems)
	}

	var outRGB []float32
	err := webgpu.WithDevice(func(dev *wgpu.Device, q *wgpu.Queue) error {
		g.dev, g.queue = dev, q
		q.WriteBuffer(g.scratch[0], 0, wgpu.ToBytes(z[:need]))

		cur := gTensor{C: c, H: latH, W: latW, Buf: 0}
		log := func(name string) {
			fmt.Printf("    VAE-GPU %-14s %dx%d×%d  (+%v)\n", name, cur.H, cur.W, cur.C, time.Since(t0).Round(time.Millisecond))
			t0 = time.Now()
		}

		var err error
		if v.PostQuant != nil {
			cur, err = g.conv(v.PostQuant, cur, 1)
			if err != nil {
				return err
			}
			log("post_quant")
		}
		cur, err = g.conv(v.ConvIn, cur, g.free(cur.Buf))
		if err != nil {
			return err
		}
		log("conv_in")

		if v.Mid != nil {
			cur, err = g.resnet(v.Mid.Res0, cur)
			if err != nil {
				return err
			}
			cur, err = g.attention(v.Mid.Attn, cur)
			if err != nil {
				return err
			}
			cur, err = g.resnet(v.Mid.Res1, cur)
			if err != nil {
				return err
			}
			log("mid_block")
		}

		for i, up := range v.UpBlocks {
			if up == nil {
				continue
			}
			for _, r := range up.Resnets {
				cur, err = g.resnet(r, cur)
				if err != nil {
					return err
				}
			}
			if up.Upsampler != nil {
				cur, err = g.upsample(cur, g.free(cur.Buf))
				if err != nil {
					return err
				}
				cur, err = g.conv(up.Upsampler, cur, g.free(cur.Buf))
				if err != nil {
					return err
				}
			}
			log(fmt.Sprintf("up_blocks.%d", i))
		}

		cur, err = g.groupNorm(v.NormOut, cur, g.free(cur.Buf))
		if err != nil {
			return err
		}
		if err := g.silu(cur); err != nil {
			return err
		}
		cur, err = g.conv(v.ConvOut, cur, g.free(cur.Buf))
		if err != nil {
			return err
		}
		log("conv_out")

		raw, err := webgpu.ReadbackF32(dev, q, g.sb(cur.Buf), g.n(cur))
		if err != nil {
			return err
		}
		// (x/2+0.5).clamp → HWC RGB
		outH, outW := cur.H, cur.W
		outRGB = make([]float32, outH*outW*3)
		for y := 0; y < outH; y++ {
			for x := 0; x < outW; x++ {
				off := (y*outW + x) * 3
				for ch := 0; ch < 3 && ch < cur.C; ch++ {
					v := raw[ch*outH*outW+y*outW+x]*0.5 + 0.5
					if v < 0 {
						v = 0
					} else if v > 1 {
						v = 1
					}
					outRGB[off+ch] = v
				}
			}
		}
		return nil
	})
	return outRGB, err
}
