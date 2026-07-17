package dense

// GemvF32SIMD computes y = W @ x for dense row-major W [outRows×inCols] using Plan-9 DotTile
// (row-parallel when large). Safe to call when simd is unavailable (falls back inside DotTile).
func GemvF32SIMD(w, x, y []float32, outRows, inCols int) {
	if outRows <= 0 || inCols <= 0 {
		return
	}
	gemvF32ParallelF32(w, x, y, outRows, inCols)
}
