package dense

import (
	"runtime"
	"sync"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/simd"
)

// gemvQ4_0ParallelF32 mirrors Lucy gemvQ4_0PackedParallelF32 — shard output rows across GOMAXPROCS.
func gemvQ4_0ParallelF32(scales []float32, packed []uint32, in, out []float32, outRows, inCols int) {
	if outRows < 64 || runtime.NumCPU() < 2 {
		gemvQ4_0RowsF32(scales, packed, in, out, 0, outRows, inCols)
		return
	}
	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = runtime.NumCPU()
	}
	if workers > outRows {
		workers = outRows
	}
	chunk := (outRows + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		o0 := w * chunk
		o1 := o0 + chunk
		if o1 > outRows {
			o1 = outRows
		}
		if o0 >= o1 {
			break
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			gemvQ4_0RowsF32(scales, packed, in, out, lo, hi, inCols)
		}(o0, o1)
	}
	wg.Wait()
}

func gemvQ4_0RowsF32(scales []float32, packed []uint32, in, out []float32, rowLo, rowHi, inCols int) {
	o := rowLo
	if inCols%32 == 0 && simd.Enabled() {
		for ; o+3 < rowHi; o += 4 {
			simd.DotQ4_0Rows4(in, scales, packed, o*inCols, inCols, out[o:o+4])
		}
	}
	for ; o < rowHi; o++ {
		out[o] = float32(simd.DotQ4_0Row(in, scales, packed, o*inCols, inCols, 0))
	}
}

func gemvQ8ParallelF32(scales []float32, qs []int8, in, out []float32, outRows, inCols int) {
	if outRows < 64 || runtime.NumCPU() < 2 {
		gemvQ8RowsF32(scales, qs, in, out, 0, outRows, inCols)
		return
	}
	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = runtime.NumCPU()
	}
	if workers > outRows {
		workers = outRows
	}
	chunk := (outRows + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		o0 := w * chunk
		o1 := o0 + chunk
		if o1 > outRows {
			o1 = outRows
		}
		if o0 >= o1 {
			break
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			gemvQ8RowsF32(scales, qs, in, out, lo, hi, inCols)
		}(o0, o1)
	}
	wg.Wait()
}

func gemvQ8RowsF32(scales []float32, qs []int8, in, out []float32, rowLo, rowHi, inCols int) {
	o := rowLo
	if inCols%32 == 0 && simd.Enabled() {
		for ; o+3 < rowHi; o += 4 {
			simd.DotQ8_0Rows4(in, scales, qs, o*inCols, inCols, out[o:o+4])
		}
	}
	for ; o < rowHi; o++ {
		out[o] = float32(simd.DotQ8_0Row(in, scales, qs, o*inCols, inCols, 0))
	}
}

func gemvQ41ParallelF32(scales, mins []float32, packed []uint32, in, out []float32, outRows, inCols int) {
	if outRows < 64 || runtime.NumCPU() < 2 {
		gemvQ41RowsF32(scales, mins, packed, in, out, 0, outRows, inCols)
		return
	}
	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = runtime.NumCPU()
	}
	if workers > outRows {
		workers = outRows
	}
	chunk := (outRows + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		o0 := w * chunk
		o1 := o0 + chunk
		if o1 > outRows {
			o1 = outRows
		}
		if o0 >= o1 {
			break
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			gemvQ41RowsF32(scales, mins, packed, in, out, lo, hi, inCols)
		}(o0, o1)
	}
	wg.Wait()
}

func gemvQ41RowsF32(scales, mins []float32, packed []uint32, in, out []float32, rowLo, rowHi, inCols int) {
	o := rowLo
	if inCols%32 == 0 && simd.Enabled() {
		for ; o+3 < rowHi; o += 4 {
			simd.DotQ4_1Rows4(in, scales, mins, packed, o*inCols, inCols, out[o:o+4])
		}
	}
	for ; o < rowHi; o++ {
		out[o] = float32(simd.DotQ4_1Row(in, scales, mins, packed, o*inCols, inCols, 0))
	}
}

func gemvQ5_0ParallelF32(scales []float32, qs []int8, in, out []float32, outRows, inCols int) {
	if outRows < 64 || runtime.NumCPU() < 2 {
		gemvQ5_0RowsF32(scales, qs, in, out, 0, outRows, inCols)
		return
	}
	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = runtime.NumCPU()
	}
	if workers > outRows {
		workers = outRows
	}
	chunk := (outRows + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		o0 := w * chunk
		o1 := o0 + chunk
		if o1 > outRows {
			o1 = outRows
		}
		if o0 >= o1 {
			break
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			gemvQ5_0RowsF32(scales, qs, in, out, lo, hi, inCols)
		}(o0, o1)
	}
	wg.Wait()
}

func gemvQ5_0RowsF32(scales []float32, qs []int8, in, out []float32, rowLo, rowHi, inCols int) {
	o := rowLo
	if inCols%32 == 0 && simd.Enabled() {
		for ; o+3 < rowHi; o += 4 {
			simd.DotQ5_0Rows4(in, scales, qs, o*inCols, inCols, out[o:o+4])
		}
	}
	for ; o < rowHi; o++ {
		out[o] = float32(simd.DotQ5_0Row(in, scales, qs, o*inCols, inCols, 0))
	}
}

func gemvQ5_1ParallelF32(scales, mins []float32, qs []int8, in, out []float32, outRows, inCols int) {
	if outRows < 64 || runtime.NumCPU() < 2 {
		gemvQ5_1RowsF32(scales, mins, qs, in, out, 0, outRows, inCols)
		return
	}
	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = runtime.NumCPU()
	}
	if workers > outRows {
		workers = outRows
	}
	chunk := (outRows + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		o0 := w * chunk
		o1 := o0 + chunk
		if o1 > outRows {
			o1 = outRows
		}
		if o0 >= o1 {
			break
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			gemvQ5_1RowsF32(scales, mins, qs, in, out, lo, hi, inCols)
		}(o0, o1)
	}
	wg.Wait()
}

func gemvQ5_1RowsF32(scales, mins []float32, qs []int8, in, out []float32, rowLo, rowHi, inCols int) {
	for o := rowLo; o < rowHi; o++ {
		out[o] = float32(simd.DotQ5_1Row(in, scales, mins, qs, o*inCols, inCols, 0))
	}
}

// writeGemvF32 runs gemv into y when T is float32 (in-place); otherwise scratch + convert.
func writeGemvF32[T core.Numeric](y []T, out int, gemv func(dst []float32)) {
	if yF, ok := any(y).([]float32); ok && len(yF) >= out {
		gemv(yF[:out])
		return
	}
	tmp := make([]float32, out)
	gemv(tmp)
	core.SliceFromFloat32(tmp, y[:out])
}
