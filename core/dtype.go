package core

import (
	"fmt"
	"strconv"
)

// DType is the native element type for tensors / weight payloads.
// Welvet executes in this type (or a QuantFormat packing of it) — no QAT façade.
// Values 0–33 match the full Welvet/poly numeric matrix (including Go-native & LLM formats).
type DType int

const (
	DTypeFloat64  DType = 0  // 64-bit double
	DTypeFloat32  DType = 1  // Standard 32-bit float
	DTypeFloat16  DType = 2  // 16-bit float
	DTypeBFloat16 DType = 3  // 16-bit Brain Float
	DTypeFP8E4M3  DType = 4  // 8-bit FP8 (E4M3)
	DTypeFP8E5M2  DType = 5  // 8-bit FP8 (E5M2)
	DTypeInt64    DType = 6  // 64-bit integer
	DTypeInt32    DType = 7  // 32-bit integer
	DTypeInt16    DType = 8  // 16-bit integer
	DTypeInt8     DType = 9  // 8-bit integer
	DTypeUint64   DType = 10 // 64-bit unsigned
	DTypeUint32   DType = 11 // 32-bit unsigned
	DTypeUint16   DType = 12 // 16-bit unsigned
	DTypeUint8    DType = 13 // 8-bit unsigned
	DTypeInt4     DType = 14 // 4-bit integer
	DTypeUint4    DType = 15 // 4-bit unsigned
	DTypeFP4      DType = 16 // 4-bit E2M1
	DTypeInt2     DType = 17 // 2-bit integer
	DTypeUint2    DType = 18 // 2-bit unsigned
	DTypeTernary  DType = 19 // 2-bit (Ternary: -1, 0, 1)
	DTypeBinary   DType = 20 // 1-bit (XNOR-Net)

	// Go native architecture-dependent & complex primitives
	DTypeInt        DType = 21 // System-native signed integer (32 or 64-bit)
	DTypeUint       DType = 22 // System-native unsigned integer (32 or 64-bit)
	DTypeUintptr    DType = 23 // Unsigned integer large enough for pointer bits
	DTypeComplex64  DType = 24 // complex64 (float32 real/imag)
	DTypeComplex128 DType = 25 // complex128 (float64 real/imag)

	// Extended quantization & modern LLM core formats
	DTypeNF4   DType = 26 // 4-bit NormalFloat (QLoRA)
	DTypeFP6   DType = 27 // 6-bit float (E3M2-style microscaling)
	DTypeInt6  DType = 28 // 6-bit integer
	DTypeUint6 DType = 29 // 6-bit unsigned
	DTypeInt5  DType = 30 // 5-bit integer
	DTypeUint5 DType = 31 // 5-bit unsigned
	DTypeInt3  DType = 32 // 3-bit integer
	DTypeUint3 DType = 33 // 3-bit unsigned
)

// AllDTypes is the full native dtype matrix Welvet must cover (34 types, 0–33).
var AllDTypes = []DType{
	DTypeFloat64, DTypeFloat32, DTypeFloat16, DTypeBFloat16,
	DTypeFP8E4M3, DTypeFP8E5M2,
	DTypeInt64, DTypeInt32, DTypeInt16, DTypeInt8,
	DTypeUint64, DTypeUint32, DTypeUint16, DTypeUint8,
	DTypeInt4, DTypeUint4, DTypeFP4,
	DTypeInt2, DTypeUint2, DTypeTernary, DTypeBinary,
	DTypeInt, DTypeUint, DTypeUintptr, DTypeComplex64, DTypeComplex128,
	DTypeNF4, DTypeFP6,
	DTypeInt6, DTypeUint6, DTypeInt5, DTypeUint5, DTypeInt3, DTypeUint3,
}

func (d DType) String() string {
	switch d {
	case DTypeFloat64:
		return "float64"
	case DTypeFloat32:
		return "float32"
	case DTypeFloat16:
		return "float16"
	case DTypeBFloat16:
		return "bfloat16"
	case DTypeFP8E4M3:
		return "fp8e4m3"
	case DTypeFP8E5M2:
		return "fp8e5m2"
	case DTypeInt64:
		return "int64"
	case DTypeInt32:
		return "int32"
	case DTypeInt16:
		return "int16"
	case DTypeInt8:
		return "int8"
	case DTypeUint64:
		return "uint64"
	case DTypeUint32:
		return "uint32"
	case DTypeUint16:
		return "uint16"
	case DTypeUint8:
		return "uint8"
	case DTypeInt4:
		return "int4"
	case DTypeUint4:
		return "uint4"
	case DTypeFP4:
		return "fp4"
	case DTypeInt2:
		return "int2"
	case DTypeUint2:
		return "uint2"
	case DTypeTernary:
		return "ternary"
	case DTypeBinary:
		return "binary"
	case DTypeInt:
		return "int"
	case DTypeUint:
		return "uint"
	case DTypeUintptr:
		return "uintptr"
	case DTypeComplex64:
		return "complex64"
	case DTypeComplex128:
		return "complex128"
	case DTypeNF4:
		return "nf4"
	case DTypeFP6:
		return "fp6"
	case DTypeInt6:
		return "int6"
	case DTypeUint6:
		return "uint6"
	case DTypeInt5:
		return "int5"
	case DTypeUint5:
		return "uint5"
	case DTypeInt3:
		return "int3"
	case DTypeUint3:
		return "uint3"
	default:
		return fmt.Sprintf("DType(%d)", int(d))
	}
}

// Bits returns storage bits per element for the dtype (before QuantFormat packing).
func (d DType) Bits() int {
	switch d {
	case DTypeComplex128:
		return 128
	case DTypeFloat64, DTypeInt64, DTypeUint64, DTypeComplex64:
		return 64
	case DTypeFloat32, DTypeInt32, DTypeUint32:
		return 32
	case DTypeFloat16, DTypeBFloat16, DTypeInt16, DTypeUint16:
		return 16
	case DTypeFP8E4M3, DTypeFP8E5M2, DTypeInt8, DTypeUint8:
		return 8
	case DTypeFP6, DTypeInt6, DTypeUint6:
		return 6
	case DTypeInt5, DTypeUint5:
		return 5
	case DTypeInt4, DTypeUint4, DTypeFP4, DTypeNF4:
		return 4
	case DTypeInt3, DTypeUint3:
		return 3
	case DTypeInt2, DTypeUint2, DTypeTernary:
		return 2
	case DTypeBinary:
		return 1
	case DTypeInt, DTypeUint:
		return strconv.IntSize
	case DTypeUintptr:
		return 32 << (^uintptr(0) >> 63) // 32 or 64
	default:
		return 32
	}
}
