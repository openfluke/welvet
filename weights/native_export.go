package weights

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/openfluke/welvet/core"
)

// EncodeNative packs float32 weights into FormatNone storage bytes for dt.
// Used by entity.Repack and other persistence paths. scale is 1 for IEEE
// floats and a max-abs scale for low-bit integer/binary packs.
func EncodeNative(dt core.DType, w []float32) (raw []byte, scale float32, err error) {
	if dt == core.DTypeFloat32 {
		return EncodeF32LE(w), 1, nil
	}
	return packNative(dt, w)
}

// DecodeNative unpacks FormatNone storage bytes back to float32.
func DecodeNative(dt core.DType, raw []byte, scale float32, n int) ([]float32, error) {
	if n <= 0 {
		return nil, fmt.Errorf("weights: DecodeNative bad n=%d", n)
	}
	if dt == core.DTypeFloat32 {
		if len(raw) < n*4 {
			return nil, fmt.Errorf("weights: DecodeNative f32 truncated")
		}
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4 : i*4+4]))
		}
		return out, nil
	}
	return unpackNative(dt, raw, scale, n)
}
