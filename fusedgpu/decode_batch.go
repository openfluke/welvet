package fusedgpu

import (
	"encoding/binary"
	"fmt"
	"runtime"
	"time"

	"github.com/openfluke/webgpu/wgpu"
)

func (e *engine) disp(pass *wgpu.ComputePassEncoder, pipe *wgpu.ComputePipeline, bg *wgpu.BindGroup, x, y, z uint32) {
	pass.SetPipeline(pipe)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(x, y, z)
}

func (e *engine) recordLayers(pass *wgpu.ComputePassEncoder) {
	m := e.m
	p := e.pipe
	qkvWG := (uint32(m.qDim+m.kvDim+m.kvDim) + 63) / 64
	for i := range e.blocks {
		tag := fmt.Sprintf("L%d", i)
		e.disp(pass, p["rmsnorm"], e.bg[tag+"_rms1"], 1, 1, 1)
		e.disp(pass, p["qkv"], e.bg[tag+"_qkv"], qkvWG, 1, 1)
		e.disp(pass, p["rope"], e.bg[tag+"_ropeq"], (uint32(m.heads)+63)/64, 1, 1)
		e.disp(pass, p["rope"], e.bg[tag+"_ropek"], (uint32(m.kvHeads)+63)/64, 1, 1)
		e.disp(pass, p["kv"], e.bg[tag+"_kv"], (uint32(m.kvDim)+63)/64, 1, 1)
		e.disp(pass, p["attn"], e.bg[tag+"_attn"], uint32(m.heads), 1, 1)
		e.disp(pass, p["q4gemv"], e.bg[tag+"_o"], (uint32(m.hidden)+63)/64, 1, 1)
		e.disp(pass, p["resid"], e.bg[tag+"_r1"], (uint32(m.hidden)+63)/64, 1, 1)

		e.disp(pass, p["rmsnorm"], e.bg[tag+"_rms2"], 1, 1, 1)
		e.disp(pass, p["swiglu"], e.bg[tag+"_sw"], (uint32(m.intermediate)+63)/64, 1, 1)
		e.disp(pass, p["q4gemv"], e.bg[tag+"_d"], (uint32(m.hidden)+63)/64, 1, 1)
		e.disp(pass, p["resid"], e.bg[tag+"_r2"], (uint32(m.hidden)+63)/64, 1, 1)
	}
	e.disp(pass, p["rmsnorm"], e.bg["fnorm"], 1, 1, 1)
}

func (e *engine) recordSample(pass *wgpu.ComputePassEncoder) {
	m := e.m
	p := e.pipe
	e.disp(pass, p["q4gemv"], e.bg["lm"], (uint32(m.vocab)+63)/64, 1, 1)
	e.disp(pass, p["argmax"], e.bg["argmax"], 1, 1, 1)
	e.disp(pass, p["advance"], e.bg["advance"], 1, 1, 1)
}

func (e *engine) recordPrefill(pass *wgpu.ComputePassEncoder, promptLen int) {
	p := e.pipe
	for i := 0; i < promptLen; i++ {
		e.disp(pass, p["embed_p"], e.bg["embed_p"], (uint32(e.m.hidden)+63)/64, 1, 1)
		e.recordLayers(pass)
		if i+1 < promptLen {
			e.disp(pass, p["inc_pos"], e.bg["inc_pos"], 1, 1, 1)
		} else {
			e.recordSample(pass)
		}
	}
}

func (e *engine) recordDecodeChunk(pass *wgpu.ComputePassEncoder, k int) {
	p := e.pipe
	for i := 0; i < k; i++ {
		e.disp(pass, p["embed"], e.bg["embed_tok"], (uint32(e.m.hidden)+63)/64, 1, 1)
		e.recordLayers(pass)
		e.recordSample(pass)
	}
}

func (e *engine) runEncoded(enc *wgpu.CommandEncoder, histCount int) ([]uint32, error) {
	bytes := uint64(histCount * 4)
	if bytes < 4 {
		bytes = 4
	}
	enc.CopyBufferToBuffer(e.histBuf, 0, e.stagingHist, 0, bytes)
	cmd, err := enc.Finish(nil)
	if err != nil {
		return nil, err
	}
	e.queue.Submit(cmd)

	done := make(chan struct{})
	var st wgpu.BufferMapAsyncStatus
	if err := e.stagingHist.MapAsync(wgpu.MapModeRead, 0, bytes, func(status wgpu.BufferMapAsyncStatus) {
		st = status
		close(done)
	}); err != nil {
		return nil, err
	}
	deadline := time.Now().Add(120 * time.Second)
	for {
		e.device.Poll(false, nil)
		select {
		case <-done:
			if st != wgpu.BufferMapAsyncStatusSuccess {
				return nil, fmt.Errorf("MapAsync %v", st)
			}
			raw := e.stagingHist.GetMappedRange(0, uint(bytes))
			out := make([]uint32, histCount)
			for i := 0; i < histCount; i++ {
				out[i] = binary.LittleEndian.Uint32(raw[i*4 : i*4+4])
			}
			e.stagingHist.Unmap()
			return out, nil
		default:
			if time.Now().After(deadline) {
				return nil, fmt.Errorf("MapAsync timeout")
			}
			runtime.Gosched()
		}
	}
}

func (e *engine) generatePrefillThenDecode(prompt []uint32, genTokens int) (out []uint32, prefillS, decodeS float64, err error) {
	if len(prompt) == 0 {
		prompt = []uint32{1}
	}
	if genTokens < 1 {
		genTokens = 1
	}

	e.queue.WriteBuffer(e.promptBuf, 0, u32Bytes(prompt))
	e.queue.WriteBuffer(e.step, 0, packU32(0, 0))

	t0 := time.Now()
	enc, err := e.device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, 0, 0, err
	}
	pass := enc.BeginComputePass(nil)
	e.recordPrefill(pass, len(prompt))
	pass.End()
	first, err := e.runEncoded(enc, 1)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("prefill: %w", err)
	}
	prefillDur := time.Since(t0)
	prefillS = float64(len(prompt)) / prefillDur.Seconds()
	out = append(out, first...)

	remain := genTokens - 1
	t1 := time.Now()
	nDec := 0
	if remain > 0 {
		e.queue.WriteBuffer(e.step, 0, packU32(uint32(len(prompt)), 0))

		enc2, err := e.device.CreateCommandEncoder(nil)
		if err != nil {
			return out, prefillS, 0, err
		}
		pass2 := enc2.BeginComputePass(nil)
		e.recordDecodeChunk(pass2, remain)
		pass2.End()
		chunk, err := e.runEncoded(enc2, remain)
		if err != nil {
			return out, prefillS, 0, fmt.Errorf("decode: %w", err)
		}
		out = append(out, chunk...)
		nDec = remain
	} else {
		nDec = 1
	}
	decodeDur := time.Since(t1)
	if remain == 0 {
		decodeDur = prefillDur
	}
	if decodeDur > 0 && nDec > 0 {
		decodeS = float64(nDec) / decodeDur.Seconds()
	}

	fmt.Printf("prefill %d tok in %v (%.1f tok/s) | decode %d tok in %v (%.1f tok/s) [1 compute pass + 1 sync]\n",
		len(prompt), prefillDur.Round(time.Millisecond), prefillS,
		nDec, decodeDur.Round(time.Millisecond), decodeS)
	return out, prefillS, decodeS, nil
}
