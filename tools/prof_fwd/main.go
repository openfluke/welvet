package main

import (
	"fmt"
	"os"
	"runtime"
	"runtime/pprof"
	"time"

	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/simd"
	"github.com/openfluke/welvet/transformer"
)

func main() {
	fmt.Println("simd", simd.Enabled(), "procs", runtime.GOMAXPROCS(0))
	path := "/home/openfluke/git/chaosglue/welvet/octo/octo_entities/HuggingFaceTB--SmolLM2-135M-Instruct-q4.entity"
	m, err := transformer.LoadEntity(path)
	if err != nil {
		panic(err)
	}
	var p transformer.ExecProfile
	for _, ep := range transformer.NamedProfiles() {
		if ep.Name == "simd_fuse" {
			p = ep
			p.PackFormat = quant.FormatQ4_0
			break
		}
	}
	if err := m.ApplyExec(p); err != nil {
		panic(err)
	}
	fmt.Println("backend", m.Exec.Backend, "fused", m.Fused, "pack", m.PackFormat)

	ids := make([]uint32, 32)
	for i := range ids {
		ids[i] = uint32(i + 1)
	}
	m.ResetKV()
	if _, err := m.ForwardTokens(ids); err != nil {
		panic(err)
	}

	f, err := os.Create("/tmp/welvet_cpu.prof")
	if err != nil {
		panic(err)
	}
	_ = pprof.StartCPUProfile(f)

	const gen = 64
	t0 := time.Now()
	var next uint32 = 20
	for i := 0; i < gen; i++ {
		logits, err := m.ForwardTokens([]uint32{next})
		if err != nil {
			panic(err)
		}
		bi := 0
		bv := logits[0]
		for j := 1; j < len(logits); j++ {
			if logits[j] > bv {
				bv = logits[j]
				bi = j
			}
		}
		next = uint32(bi)
	}
	dt := time.Since(t0)
	pprof.StopCPUProfile()
	_ = f.Close()
	fmt.Printf("decode %d toks in %v → %.2f tok/s\n", gen, dt, float64(gen)/dt.Seconds())
	fmt.Println("profile: /tmp/welvet_cpu.prof")
}
