package webgpu

import (
	"fmt"
	"math"

	"github.com/openfluke/webgpu/wgpu"
	"github.com/openfluke/welvet/tiling"
)

type rnnParams struct {
	BatchSize  uint32
	InputSize  uint32
	HiddenSize uint32
	Padding    uint32
}

type lstmParams struct {
	BatchSize  uint32
	InputSize  uint32
	HiddenSize uint32
	Padding    uint32
}

func storageBytes(n int) uint64 {
	sz := uint64(n * 4)
	if sz < 64 {
		return 64
	}
	return sz
}

func (s *session) ensureRNNStepPipe() error {
	if s.pipeRNNStep != nil {
		return nil
	}
	var err error
	s.pipeRNNStep, err = makePipeline(s.device, ShaderRNNStepPreAct, "welvet-rnn-step")
	return err
}

func (s *session) ensureLSTMStepPipe() error {
	if s.pipeLSTMStep != nil {
		return nil
	}
	var err error
	s.pipeLSTMStep, err = makePipeline(s.device, ShaderLSTMStepPreAct, "welvet-lstm-step")
	return err
}

func (s *session) ensureRNNBwdDXPipe(tileSize int) (*wgpu.ComputePipeline, error) {
	if tileSize <= 0 {
		tileSize = 64
	}
	if s.rnnBwdDXPipes == nil {
		s.rnnBwdDXPipes = make(map[int]*wgpu.ComputePipeline)
	}
	if p, ok := s.rnnBwdDXPipes[tileSize]; ok {
		return p, nil
	}
	p, err := makePipeline(s.device, shaderTiledRNNBackwardDX(tileSize), fmt.Sprintf("welvet-rnn-bwd-dx-%d", tileSize))
	if err != nil {
		return nil, err
	}
	s.rnnBwdDXPipes[tileSize] = p
	return p, nil
}

func (s *session) ensureRNNBwdDWPipe(tileSize int) (*wgpu.ComputePipeline, error) {
	if tileSize <= 0 {
		tileSize = 64
	}
	if s.rnnBwdDWPipes == nil {
		s.rnnBwdDWPipes = make(map[int]*wgpu.ComputePipeline)
	}
	if p, ok := s.rnnBwdDWPipes[tileSize]; ok {
		return p, nil
	}
	p, err := makePipeline(s.device, shaderTiledRNNBackwardDW(tileSize), fmt.Sprintf("welvet-rnn-bwd-dw-%d", tileSize))
	if err != nil {
		return nil, err
	}
	s.rnnBwdDWPipes[tileSize] = p
	return p, nil
}

func (s *session) ensureLSTMBwdDXPipe(tileSize int) (*wgpu.ComputePipeline, error) {
	if tileSize <= 0 {
		tileSize = 64
	}
	if s.lstmBwdDXPipes == nil {
		s.lstmBwdDXPipes = make(map[int]*wgpu.ComputePipeline)
	}
	if p, ok := s.lstmBwdDXPipes[tileSize]; ok {
		return p, nil
	}
	p, err := makePipeline(s.device, shaderTiledLSTMBackwardDX(tileSize), fmt.Sprintf("welvet-lstm-bwd-dx-%d", tileSize))
	if err != nil {
		return nil, err
	}
	s.lstmBwdDXPipes[tileSize] = p
	return p, nil
}

func (s *session) ensureLSTMBwdDWPipe(tileSize int) (*wgpu.ComputePipeline, error) {
	if tileSize <= 0 {
		tileSize = 64
	}
	if s.lstmBwdDWPipes == nil {
		s.lstmBwdDWPipes = make(map[int]*wgpu.ComputePipeline)
	}
	if p, ok := s.lstmBwdDWPipes[tileSize]; ok {
		return p, nil
	}
	p, err := makePipeline(s.device, shaderTiledLSTMBackwardDW(tileSize), fmt.Sprintf("welvet-lstm-bwd-dw-%d", tileSize))
	if err != nil {
		return nil, err
	}
	s.lstmBwdDWPipes[tileSize] = p
	return p, nil
}

// RNNForwardSeq runs fused RNN steps on device for every timestep.
//
// input [batch,seq,in], weights [ih+hh+bias], pre/post [batch,seq,hid] (pre=linear, post=tanh).
func RNNForwardSeq(input, weights, pre, post []float32, batch, seq, in, hid int) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu RNNForwardSeq: %w", initErr)
	}
	needIn := batch * seq * in
	needOut := batch * seq * hid
	wN := hid*in + hid*hid + hid
	if len(input) < needIn || len(weights) < wN || len(pre) < needOut || len(post) < needOut {
		return fmt.Errorf("webgpu RNNForwardSeq: shape")
	}
	if err := sess.ensureRNNStepPipe(); err != nil {
		return err
	}
	return sess.rnnForwardSeq(input[:needIn], weights[:wN], pre[:needOut], post[:needOut], batch, seq, in, hid)
}

// RNNBackwardSeq runs BPTT with on-device DX/DW per step (FormatNone path).
func RNNBackwardSeq(gradOut, input, post, weights, gradIn, gradW []float32, batch, seq, in, hid, tileSize int) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu RNNBackwardSeq: %w", initErr)
	}
	needIn := batch * seq * in
	needOut := batch * seq * hid
	wN := hid*in + hid*hid + hid
	if len(gradOut) < needOut || len(input) < needIn || len(post) < needOut ||
		len(weights) < wN || len(gradIn) < needIn || len(gradW) < wN {
		return fmt.Errorf("webgpu RNNBackwardSeq: shape")
	}
	if _, err := sess.ensureRNNBwdDXPipe(tileSize); err != nil {
		return err
	}
	if _, err := sess.ensureRNNBwdDWPipe(tileSize); err != nil {
		return err
	}
	return sess.rnnBackwardSeq(gradOut[:needOut], input[:needIn], post[:needOut], weights[:wN],
		gradIn[:needIn], gradW[:wN], batch, seq, in, hid, tileSize)
}

// LSTMForwardSeq runs fused LSTM steps on device; pre is [batch,seq,5*hid].
func LSTMForwardSeq(input, weights, pre, post []float32, batch, seq, in, hid int) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu LSTMForwardSeq: %w", initErr)
	}
	needIn := batch * seq * in
	needPre := batch * seq * 5 * hid
	needPost := batch * seq * hid
	wN := 4 * (hid*in + hid*hid + hid)
	if len(input) < needIn || len(weights) < wN || len(pre) < needPre || len(post) < needPost {
		return fmt.Errorf("webgpu LSTMForwardSeq: shape")
	}
	if err := sess.ensureLSTMStepPipe(); err != nil {
		return err
	}
	return sess.lstmForwardSeq(input[:needIn], weights[:wN], pre[:needPre], post[:needPost], batch, seq, in, hid)
}

// LSTMBackwardSeq runs BPTT with on-device DX/DW per step; pre from forward.
func LSTMBackwardSeq(gradOut, input, pre, weights, gradIn, gradW []float32, batch, seq, in, hid, tileSize int) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu LSTMBackwardSeq: %w", initErr)
	}
	needIn := batch * seq * in
	needPre := batch * seq * 5 * hid
	needOut := batch * seq * hid
	wN := 4 * (hid*in + hid*hid + hid)
	if len(gradOut) < needOut || len(input) < needIn || len(pre) < needPre ||
		len(weights) < wN || len(gradIn) < needIn || len(gradW) < wN {
		return fmt.Errorf("webgpu LSTMBackwardSeq: shape")
	}
	if _, err := sess.ensureLSTMBwdDXPipe(tileSize); err != nil {
		return err
	}
	if _, err := sess.ensureLSTMBwdDWPipe(tileSize); err != nil {
		return err
	}
	return sess.lstmBackwardSeq(gradOut[:needOut], input[:needIn], pre[:needPre], weights[:wN],
		gradIn[:needIn], gradW[:wN], batch, seq, in, hid, tileSize)
}

func (s *session) rnnForwardSeq(input, weights, pre, post []float32, batch, seq, in, hid int) error {
	ihN, hhN := hid*in, hid*hid
	wIH := weights[:ihN]
	wHH := weights[ihN : ihN+hhN]
	bias := weights[ihN+hhN:]

	hPrev := make([]float32, batch*hid)
	hCurr := make([]float32, batch*hid)
	stepPre := make([]float32, batch*hid)
	xt := make([]float32, batch*in)

	for t := 0; t < seq; t++ {
		for b := 0; b < batch; b++ {
			copy(xt[b*in:(b+1)*in], input[b*seq*in+t*in:b*seq*in+(t+1)*in])
		}
		if err := s.rnnStep(xt, hPrev, wIH, wHH, bias, stepPre, hCurr, batch, in, hid); err != nil {
			return fmt.Errorf("rnn fwd t=%d: %w", t, err)
		}
		for b := 0; b < batch; b++ {
			off := b*seq*hid + t*hid
			copy(pre[off:off+hid], stepPre[b*hid:(b+1)*hid])
			copy(post[off:off+hid], hCurr[b*hid:(b+1)*hid])
		}
		copy(hPrev, hCurr)
	}
	return nil
}

func (s *session) rnnStep(input, hPrev, wIH, wHH, bias, preAct, hCurr []float32, batch, in, hid int) error {
	dev, q := s.device, s.queue
	n := batch * hid

	inBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-rnn-in", Contents: wgpu.ToBytes(input),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer inBuf.Destroy()
	hpBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-rnn-hprev", Contents: wgpu.ToBytes(hPrev),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer hpBuf.Destroy()
	wIHBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-rnn-wih", Contents: wgpu.ToBytes(wIH),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer wIHBuf.Destroy()
	wHHBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-rnn-whh", Contents: wgpu.ToBytes(wHH),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer wHHBuf.Destroy()
	biasBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-rnn-bias", Contents: wgpu.ToBytes(bias),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer biasBuf.Destroy()

	preBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "welvet-rnn-pre", Size: storageBytes(n),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer preBuf.Destroy()
	hBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "welvet-rnn-h", Size: storageBytes(n),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer hBuf.Destroy()

	p := rnnParams{BatchSize: uint32(batch), InputSize: uint32(in), HiddenSize: uint32(hid)}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-rnn-p", Contents: wgpu.ToBytes([]rnnParams{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()

	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeRNNStep.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: inBuf, Offset: 0, Size: inBuf.GetSize()},
			{Binding: 2, Buffer: hpBuf, Offset: 0, Size: hpBuf.GetSize()},
			{Binding: 3, Buffer: wIHBuf, Offset: 0, Size: wIHBuf.GetSize()},
			{Binding: 4, Buffer: wHHBuf, Offset: 0, Size: wHHBuf.GetSize()},
			{Binding: 5, Buffer: biasBuf, Offset: 0, Size: biasBuf.GetSize()},
			{Binding: 6, Buffer: preBuf, Offset: 0, Size: preBuf.GetSize()},
			{Binding: 7, Buffer: hBuf, Offset: 0, Size: hBuf.GetSize()},
		},
	})
	if err != nil {
		return err
	}
	defer bg.Release()

	if err := s.dispatch1D(s.pipeRNNStep, bg, tiling.GPUWorkgroupsX(hid, 64), uint32(batch)); err != nil {
		return err
	}

	outPre, err := readbackF32(dev, q, preBuf, n)
	if err != nil {
		return err
	}
	outH, err := readbackF32(dev, q, hBuf, n)
	if err != nil {
		return err
	}
	copy(preAct, outPre)
	copy(hCurr, outH)
	return nil
}

func (s *session) rnnBackwardSeq(gradOut, input, post, weights, gradIn, gradW []float32, batch, seq, in, hid, tileSize int) error {
	ihN := hid * in
	wIH := weights[:ihN]
	wHH := weights[ihN : ihN+hid*hid]

	clearF32(gradW)
	clearF32(gradIn)

	gH := make([]float32, batch*hid)
	gComb := make([]float32, batch*hid)
	gPre := make([]float32, batch*hid)
	xt := make([]float32, batch*in)
	hPrev := make([]float32, batch*hid)
	postT := make([]float32, batch*hid)
	stepDX := make([]float32, batch*in)
	stepDW := make([]float32, len(weights))
	for t := seq - 1; t >= 0; t-- {
		for b := 0; b < batch; b++ {
			off := b*seq*hid + t*hid
			for h := 0; h < hid; h++ {
				i := b*hid + h
				gComb[i] = gH[i] + gradOut[off+h]
			}
			copy(postT[b*hid:(b+1)*hid], post[off:off+hid])
			copy(xt[b*in:(b+1)*in], input[b*seq*in+t*in:b*seq*in+(t+1)*in])
			if t == 0 {
				clear(hPrev[b*hid : (b+1)*hid])
			} else {
				copy(hPrev[b*hid:(b+1)*hid], post[b*seq*hid+(t-1)*hid:b*seq*hid+t*hid])
			}
		}
		for i := range stepDW {
			stepDW[i] = 0
		}
		if err := s.rnnBackwardStep(gComb, xt, postT, hPrev, wIH, stepDX, stepDW, batch, in, hid, tileSize); err != nil {
			return fmt.Errorf("rnn bwd t=%d: %w", t, err)
		}
		for b := 0; b < batch; b++ {
			copy(gradIn[b*seq*in+t*in:b*seq*in+(t+1)*in], stepDX[b*in:(b+1)*in])
		}
		for i := range stepDW {
			gradW[i] += stepDW[i]
		}
		for b := 0; b < batch; b++ {
			off := b*seq*hid + t*hid
			for h := 0; h < hid; h++ {
				i := b*hid + h
				hc := post[off+h]
				gPre[i] = gComb[i] * (1 - hc*hc)
			}
		}
		ghNext := make([]float32, batch*hid)
		if err := DenseGEMVT(wHH, gPre, ghNext, batch, hid, hid); err != nil {
			return fmt.Errorf("rnn bwd hh gemvt t=%d: %w", t, err)
		}
		copy(gH, ghNext)
	}
	return nil
}

func (s *session) rnnBackwardStep(gradComb, input, hCurr, hPrev, wIH, gradIn, gradW []float32, batch, in, hid, tileSize int) error {
	if tileSize <= 0 {
		tileSize = 64
	}
	pipeDX, err := s.ensureRNNBwdDXPipe(tileSize)
	if err != nil {
		return err
	}
	pipeDW, err := s.ensureRNNBwdDWPipe(tileSize)
	if err != nil {
		return err
	}
	dev, q := s.device, s.queue

	gBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-rnn-bwd-g", Contents: wgpu.ToBytes(gradComb),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer gBuf.Destroy()
	wIHBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-rnn-bwd-wih", Contents: wgpu.ToBytes(wIH),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer wIHBuf.Destroy()
	hcBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-rnn-bwd-hc", Contents: wgpu.ToBytes(hCurr),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer hcBuf.Destroy()
	inBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-rnn-bwd-in", Contents: wgpu.ToBytes(input),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer inBuf.Destroy()
	hpBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-rnn-bwd-hp", Contents: wgpu.ToBytes(hPrev),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer hpBuf.Destroy()

	dxBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "welvet-rnn-bwd-dx", Size: storageBytes(batch * in),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer dxBuf.Destroy()
	dwBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-rnn-bwd-dw", Contents: wgpu.ToBytes(gradW),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer dwBuf.Destroy()

	p := rnnParams{BatchSize: uint32(batch), InputSize: uint32(in), HiddenSize: uint32(hid)}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-rnn-bwd-p", Contents: wgpu.ToBytes([]rnnParams{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()

	bgDX, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: pipeDX.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: gBuf, Offset: 0, Size: gBuf.GetSize()},
			{Binding: 2, Buffer: wIHBuf, Offset: 0, Size: wIHBuf.GetSize()},
			{Binding: 3, Buffer: hcBuf, Offset: 0, Size: hcBuf.GetSize()},
			{Binding: 4, Buffer: dxBuf, Offset: 0, Size: dxBuf.GetSize()},
		},
	})
	if err != nil {
		return err
	}
	defer bgDX.Release()

	bgDW, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: pipeDW.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: gBuf, Offset: 0, Size: gBuf.GetSize()},
			{Binding: 2, Buffer: inBuf, Offset: 0, Size: inBuf.GetSize()},
			{Binding: 3, Buffer: hcBuf, Offset: 0, Size: hcBuf.GetSize()},
			{Binding: 4, Buffer: hpBuf, Offset: 0, Size: hpBuf.GetSize()},
			{Binding: 5, Buffer: dwBuf, Offset: 0, Size: dwBuf.GetSize()},
		},
	})
	if err != nil {
		return err
	}
	defer bgDW.Release()

	enc, err := dev.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeDX)
	pass.SetBindGroup(0, bgDX, nil)
	pass.DispatchWorkgroups(tiling.GPUWorkgroupsX(in, tileSize), uint32(batch), 1)
	pass.SetPipeline(pipeDW)
	pass.SetBindGroup(0, bgDW, nil)
	pass.DispatchWorkgroups(tiling.GPUWorkgroupsX(hid, tileSize), 1, 1)
	pass.End()
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	q.Submit(cmd)

	outDX, err := readbackF32(dev, q, dxBuf, batch*in)
	if err != nil {
		return err
	}
	copy(gradIn, outDX)
	outDW, err := readbackF32(dev, q, dwBuf, len(gradW))
	if err != nil {
		return err
	}
	copy(gradW, outDW)
	return nil
}

func (s *session) lstmForwardSeq(input, weights, pre, post []float32, batch, seq, in, hid int) error {
	hPrev := make([]float32, batch*hid)
	cPrev := make([]float32, batch*hid)
	hCurr := make([]float32, batch*hid)
	cCurr := make([]float32, batch*hid)
	stepPre := make([]float32, batch*5*hid)
	xt := make([]float32, batch*in)

	for t := 0; t < seq; t++ {
		for b := 0; b < batch; b++ {
			copy(xt[b*in:(b+1)*in], input[b*seq*in+t*in:b*seq*in+(t+1)*in])
		}
		if err := s.lstmStep(xt, hPrev, cPrev, weights, stepPre, hCurr, cCurr, batch, in, hid); err != nil {
			return fmt.Errorf("lstm fwd t=%d: %w", t, err)
		}
		for b := 0; b < batch; b++ {
			pOff := b*seq*5*hid + t*5*hid
			oOff := b*seq*hid + t*hid
			copy(pre[pOff:pOff+5*hid], stepPre[b*5*hid:(b+1)*5*hid])
			copy(post[oOff:oOff+hid], hCurr[b*hid:(b+1)*hid])
		}
		copy(hPrev, hCurr)
		copy(cPrev, cCurr)
	}
	return nil
}

func (s *session) lstmStep(input, hPrev, cPrev, weights, preAct, hCurr, cCurr []float32, batch, in, hid int) error {
	dev, q := s.device, s.queue
	n := batch * hid
	pN := batch * 5 * hid

	inBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-lstm-in", Contents: wgpu.ToBytes(input),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer inBuf.Destroy()
	hpBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-lstm-hprev", Contents: wgpu.ToBytes(hPrev),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer hpBuf.Destroy()
	cpBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-lstm-cprev", Contents: wgpu.ToBytes(cPrev),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer cpBuf.Destroy()
	wBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-lstm-w", Contents: wgpu.ToBytes(weights),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer wBuf.Destroy()

	hBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "welvet-lstm-h", Size: storageBytes(n),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer hBuf.Destroy()
	cBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "welvet-lstm-c", Size: storageBytes(n),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer cBuf.Destroy()
	pBufOut, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "welvet-lstm-preact", Size: storageBytes(pN),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBufOut.Destroy()

	p := lstmParams{BatchSize: uint32(batch), InputSize: uint32(in), HiddenSize: uint32(hid)}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-lstm-p", Contents: wgpu.ToBytes([]lstmParams{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()

	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeLSTMStep.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: inBuf, Offset: 0, Size: inBuf.GetSize()},
			{Binding: 2, Buffer: hpBuf, Offset: 0, Size: hpBuf.GetSize()},
			{Binding: 3, Buffer: cpBuf, Offset: 0, Size: cpBuf.GetSize()},
			{Binding: 4, Buffer: wBuf, Offset: 0, Size: wBuf.GetSize()},
			{Binding: 5, Buffer: hBuf, Offset: 0, Size: hBuf.GetSize()},
			{Binding: 6, Buffer: cBuf, Offset: 0, Size: cBuf.GetSize()},
			{Binding: 7, Buffer: pBufOut, Offset: 0, Size: pBufOut.GetSize()},
		},
	})
	if err != nil {
		return err
	}
	defer bg.Release()

	if err := s.dispatch1D(s.pipeLSTMStep, bg, tiling.GPUWorkgroupsX(hid, 64), uint32(batch)); err != nil {
		return err
	}

	outP, err := readbackF32(dev, q, pBufOut, pN)
	if err != nil {
		return err
	}
	outH, err := readbackF32(dev, q, hBuf, n)
	if err != nil {
		return err
	}
	outC, err := readbackF32(dev, q, cBuf, n)
	if err != nil {
		return err
	}
	copy(preAct, outP)
	copy(hCurr, outH)
	copy(cCurr, outC)
	return nil
}

func (s *session) lstmBackwardSeq(gradOut, input, pre, weights, gradIn, gradW []float32, batch, seq, in, hid, tileSize int) error {
	gateN := hid*in + hid*hid + hid
	ihN := hid * in

	clearF32(gradW)
	clearF32(gradIn)

	gH := make([]float32, batch*hid)
	gC := make([]float32, batch*hid)
	xt := make([]float32, batch*in)
	hPrev := make([]float32, batch*hid)
	cPrev := make([]float32, batch*hid)
	stepPre := make([]float32, batch*5*hid)
	gradOutT := make([]float32, batch*hid)
	stepDX := make([]float32, batch*in)
	stepDW := make([]float32, len(weights))
	di := make([]float32, batch*hid)
	df := make([]float32, batch*hid)
	dg := make([]float32, batch*hid)
	do := make([]float32, batch*hid)
	nextGC := make([]float32, batch*hid)
	nextGH := make([]float32, batch*hid)
	ghPart := make([]float32, batch*hid)

	for t := seq - 1; t >= 0; t-- {
		for b := 0; b < batch; b++ {
			pOff := b*seq*5*hid + t*5*hid
			oOff := b*seq*hid + t*hid
			copy(stepPre[b*5*hid:(b+1)*5*hid], pre[pOff:pOff+5*hid])
			copy(gradOutT[b*hid:(b+1)*hid], gradOut[oOff:oOff+hid])
			copy(xt[b*in:(b+1)*in], input[b*seq*in+t*in:b*seq*in+(t+1)*in])
			if t == 0 {
				clear(hPrev[b*hid : (b+1)*hid])
				clear(cPrev[b*hid : (b+1)*hid])
			} else {
				pP := b*seq*5*hid + (t-1)*5*hid
				copy(cPrev[b*hid:(b+1)*hid], pre[pP+4*hid:pP+5*hid])
				copy(hPrev[b*hid:(b+1)*hid], hiddenFromPreGate(pre[pP:pP+5*hid], hid))
			}
		}
		for i := range stepDW {
			stepDW[i] = 0
		}
		if err := s.lstmBackwardStep(gradOutT, gH, gC, cPrev, xt, hPrev, stepPre, weights, stepDX, stepDW, batch, in, hid, tileSize); err != nil {
			return fmt.Errorf("lstm bwd t=%d: %w", t, err)
		}
		for b := 0; b < batch; b++ {
			copy(gradIn[b*seq*in+t*in:b*seq*in+(t+1)*in], stepDX[b*in:(b+1)*in])
		}
		for i := range stepDW {
			gradW[i] += stepDW[i]
		}

		clearF32(nextGC)
		clearF32(nextGH)
		for b := 0; b < batch; b++ {
			pBase := b * 5 * hid
			for h := 0; h < hid; h++ {
				i := b*hid + h
				iS := stepPre[pBase+h]
				fS := stepPre[pBase+hid+h]
				gS := stepPre[pBase+2*hid+h]
				oS := stepPre[pBase+3*hid+h]
				cC := stepPre[pBase+4*hid+h]
				iG := sigmoidF32(iS)
				fG := sigmoidF32(fS)
				oG := sigmoidF32(oS)
				gG := float32Tanh(gS)
				cT := float32Tanh(cC)
				dh := gH[i] + gradOutT[i]
				dc := gC[i] + dh*oG*(1-cT*cT)
				nextGC[i] = dc * fG
				di[i] = dc * gG * iG * (1 - iG)
				df[i] = dc * cPrev[i] * fG * (1 - fG)
				dg[i] = dc * iG * (1 - gG*gG)
				do[i] = dh * cT * oG * (1 - oG)
			}
		}
		for g, delta := range [][]float32{di, df, dg, do} {
			off := g * gateN
			wHH := weights[off+ihN : off+ihN+hid*hid]
			clearF32(ghPart)
			if err := DenseGEMVT(wHH, delta, ghPart, batch, hid, hid); err != nil {
				return fmt.Errorf("lstm bwd hh g=%d t=%d: %w", g, t, err)
			}
			for i := range nextGH {
				nextGH[i] += ghPart[i]
			}
		}
		copy(gH, nextGH)
		copy(gC, nextGC)
	}
	return nil
}

func (s *session) lstmBackwardStep(gradOut, gradHidden, gradCell, cPrev, input, hPrev, preAct, weights, gradIn, gradW []float32, batch, in, hid, tileSize int) error {
	if tileSize <= 0 {
		tileSize = 64
	}
	pipeDX, err := s.ensureLSTMBwdDXPipe(tileSize)
	if err != nil {
		return err
	}
	pipeDW, err := s.ensureLSTMBwdDWPipe(tileSize)
	if err != nil {
		return err
	}
	dev, q := s.device, s.queue

	mk := func(label string, data []float32) (*wgpu.Buffer, error) {
		return dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label: label, Contents: wgpu.ToBytes(data),
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
		})
	}

	goBuf, err := mk("welvet-lstm-bwd-go", gradOut)
	if err != nil {
		return err
	}
	defer goBuf.Destroy()
	ghBuf, err := mk("welvet-lstm-bwd-gh", gradHidden)
	if err != nil {
		return err
	}
	defer ghBuf.Destroy()
	gcBuf, err := mk("welvet-lstm-bwd-gc", gradCell)
	if err != nil {
		return err
	}
	defer gcBuf.Destroy()
	cpBuf, err := mk("welvet-lstm-bwd-cp", cPrev)
	if err != nil {
		return err
	}
	defer cpBuf.Destroy()
	wBuf, err := mk("welvet-lstm-bwd-w", weights)
	if err != nil {
		return err
	}
	defer wBuf.Destroy()
	paBuf, err := mk("welvet-lstm-bwd-pa", preAct)
	if err != nil {
		return err
	}
	defer paBuf.Destroy()
	inBuf, err := mk("welvet-lstm-bwd-in", input)
	if err != nil {
		return err
	}
	defer inBuf.Destroy()
	hpBuf, err := mk("welvet-lstm-bwd-hp", hPrev)
	if err != nil {
		return err
	}
	defer hpBuf.Destroy()

	dxBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "welvet-lstm-bwd-dx", Size: storageBytes(batch * in),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer dxBuf.Destroy()
	dwBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-lstm-bwd-dw", Contents: wgpu.ToBytes(gradW),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer dwBuf.Destroy()

	p := lstmParams{BatchSize: uint32(batch), InputSize: uint32(in), HiddenSize: uint32(hid)}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-lstm-bwd-p", Contents: wgpu.ToBytes([]lstmParams{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()

	bgDX, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: pipeDX.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: goBuf, Offset: 0, Size: goBuf.GetSize()},
			{Binding: 2, Buffer: ghBuf, Offset: 0, Size: ghBuf.GetSize()},
			{Binding: 3, Buffer: gcBuf, Offset: 0, Size: gcBuf.GetSize()},
			{Binding: 4, Buffer: cpBuf, Offset: 0, Size: cpBuf.GetSize()},
			{Binding: 5, Buffer: wBuf, Offset: 0, Size: wBuf.GetSize()},
			{Binding: 6, Buffer: paBuf, Offset: 0, Size: paBuf.GetSize()},
			{Binding: 7, Buffer: dxBuf, Offset: 0, Size: dxBuf.GetSize()},
		},
	})
	if err != nil {
		return err
	}
	defer bgDX.Release()

	bgDW, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: pipeDW.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: goBuf, Offset: 0, Size: goBuf.GetSize()},
			{Binding: 2, Buffer: ghBuf, Offset: 0, Size: ghBuf.GetSize()},
			{Binding: 3, Buffer: gcBuf, Offset: 0, Size: gcBuf.GetSize()},
			{Binding: 4, Buffer: cpBuf, Offset: 0, Size: cpBuf.GetSize()},
			{Binding: 5, Buffer: inBuf, Offset: 0, Size: inBuf.GetSize()},
			{Binding: 6, Buffer: paBuf, Offset: 0, Size: paBuf.GetSize()},
			{Binding: 7, Buffer: hpBuf, Offset: 0, Size: hpBuf.GetSize()},
			{Binding: 8, Buffer: dwBuf, Offset: 0, Size: dwBuf.GetSize()},
		},
	})
	if err != nil {
		return err
	}
	defer bgDW.Release()

	enc, err := dev.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeDX)
	pass.SetBindGroup(0, bgDX, nil)
	pass.DispatchWorkgroups(tiling.GPUWorkgroupsX(in, tileSize), uint32(batch), 1)
	pass.SetPipeline(pipeDW)
	pass.SetBindGroup(0, bgDW, nil)
	pass.DispatchWorkgroups(tiling.GPUWorkgroupsX(hid, tileSize), 1, 1)
	pass.End()
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	q.Submit(cmd)

	outDX, err := readbackF32(dev, q, dxBuf, batch*in)
	if err != nil {
		return err
	}
	copy(gradIn, outDX)
	outDW, err := readbackF32(dev, q, dwBuf, len(gradW))
	if err != nil {
		return err
	}
	copy(gradW, outDW)
	return nil
}

func (s *session) dispatch1D(pipe *wgpu.ComputePipeline, bg *wgpu.BindGroup, wgX, wgY uint32) error {
	dev, q := s.device, s.queue
	enc, err := dev.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipe)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(wgX, wgY, 1)
	pass.End()
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	q.Submit(cmd)
	return nil
}

func clearF32(s []float32) {
	for i := range s {
		s[i] = 0
	}
}

func sigmoidF32(x float32) float32 {
	return 1 / (1 + float32(math.Exp(-float64(x))))
}

func float32Tanh(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}

func hiddenFromPreGate(p []float32, hid int) []float32 {
	out := make([]float32, hid)
	for h := 0; h < hid; h++ {
		oG := sigmoidF32(p[3*hid+h])
		out[h] = oG * float32Tanh(p[4*hid+h])
	}
	return out
}
