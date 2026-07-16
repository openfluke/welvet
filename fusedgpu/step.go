package fusedgpu

import (
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"time"

	"github.com/openfluke/webgpu/wgpu"
)

func (e *engine) resetState() {
	if e == nil || e.step == nil {
		return
	}
	e.queue.WriteBuffer(e.step, 0, packU32(0, 0))
	e.pos = 0
}

func (e *engine) release() {
	if e == nil {
		return
	}
	for _, bg := range e.bg {
		if bg != nil {
			bg.Release()
		}
	}
	e.bg = nil
	for _, p := range e.pipe {
		if p != nil {
			p.Release()
		}
	}
	e.pipe = nil
	for _, b := range e.owned {
		if b != nil {
			b.Release()
		}
	}
	e.owned = nil
	e.blocks = nil
	e.embed, e.finalNorm, e.lmScales, e.lmW = nil, nil, nil, nil
	e.step, e.token, e.promptBuf, e.histBuf, e.stagingHist = nil, nil, nil, nil, nil
	e.hidden, e.normed, e.qkvBuf, e.attnOut = nil, nil, nil, nil
	e.inter, e.logits, e.outTok, e.stagingLogits = nil, nil, nil, nil
	e.uGemvQDimH, e.uGemvHInter, e.uGemvVocabH = nil, nil, nil
	e.uQKV, e.uSwiglu, e.uRMS, e.uResidH = nil, nil, nil, nil
	e.uRopeQ, e.uRopeK, e.uAttn, e.uKV = nil, nil, nil, nil
	e.uEmbed, e.uArgMax = nil, nil

	if e.queue != nil {
		e.queue.Release()
		e.queue = nil
	}
	if e.device != nil {
		e.device.Poll(true, nil) // flush outstanding work before drop
		e.device.Release()
		e.device = nil
	}
	if e.adapter != nil {
		e.adapter.Release()
		e.adapter = nil
	}
	if e.instance != nil {
		e.instance.Release()
		e.instance = nil
	}
	e.m = nil
	runtime.GC()
}

func (e *engine) appendTokens(ids []uint32) ([]float32, error) {
	if len(ids) == 0 {
		return nil, fmt.Errorf("fusedgpu: empty ids")
	}
	logits := make([]float32, e.m.vocab)
	for i, id := range ids {
		if err := e.stepToken(id, i == len(ids)-1, logits); err != nil {
			return nil, err
		}
	}
	return logits, nil
}

func (e *engine) stepToken(id uint32, wantLogits bool, logits []float32) error {
	e.queue.WriteBuffer(e.step, 0, packU32(uint32(e.pos), 0))
	e.queue.WriteBuffer(e.token, 0, packU32(id))

	enc, err := e.device.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	pass := enc.BeginComputePass(nil)
	e.disp(pass, e.pipe["embed"], e.bg["embed_tok"], (uint32(e.m.hidden)+63)/64, 1, 1)
	e.recordLayers(pass)
	if wantLogits {
		e.disp(pass, e.pipe["q4gemv"], e.bg["lm"], (uint32(e.m.vocab)+63)/64, 1, 1)
	}
	e.disp(pass, e.pipe["inc_pos"], e.bg["inc_pos"], 1, 1, 1)
	pass.End()

	if wantLogits {
		bytes := uint64(e.m.vocab * 4)
		enc.CopyBufferToBuffer(e.logits, 0, e.stagingLogits, 0, bytes)
		cmd, err := enc.Finish(nil)
		if err != nil {
			return err
		}
		e.queue.Submit(cmd)
		if err := e.readLogits(logits); err != nil {
			return err
		}
	} else {
		cmd, err := enc.Finish(nil)
		if err != nil {
			return err
		}
		e.queue.Submit(cmd)
	}
	e.pos++
	return nil
}

func (e *engine) readLogits(dst []float32) error {
	bytes := uint64(len(dst) * 4)
	done := make(chan struct{})
	var st wgpu.BufferMapAsyncStatus
	if err := e.stagingLogits.MapAsync(wgpu.MapModeRead, 0, bytes, func(status wgpu.BufferMapAsyncStatus) {
		st = status
		close(done)
	}); err != nil {
		return err
	}
	deadline := time.Now().Add(120 * time.Second)
	for {
		e.device.Poll(false, nil)
		select {
		case <-done:
			if st != wgpu.BufferMapAsyncStatusSuccess {
				return fmt.Errorf("fusedgpu MapAsync %v", st)
			}
			raw := e.stagingLogits.GetMappedRange(0, uint(bytes))
			for i := range dst {
				dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4 : i*4+4]))
			}
			e.stagingLogits.Unmap()
			return nil
		default:
			if time.Now().After(deadline) {
				return fmt.Errorf("fusedgpu MapAsync timeout")
			}
			runtime.Gosched()
		}
	}
}
