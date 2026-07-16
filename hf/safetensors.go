package hf

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"unsafe"
)

// LoadSafetensorsSelective reads one safetensors file and decodes only tensors accepted by keep.
func LoadSafetensorsSelective(filepath string, keep func(string) bool) (map[string][]float32, error) {
	f, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("open safetensors: %w", err)
	}
	defer f.Close()

	var lenBuf [8]byte
	if _, err := io.ReadFull(f, lenBuf[:]); err != nil {
		return nil, fmt.Errorf("read header size: %w", err)
	}
	headerSize := binary.LittleEndian.Uint64(lenBuf[:])
	if headerSize > 256<<20 {
		return nil, fmt.Errorf("safetensors header size unreasonable: %d", headerSize)
	}
	headerBytes := make([]byte, headerSize)
	if _, err := io.ReadFull(f, headerBytes); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}

	var rawHeader map[string]any
	if err := json.Unmarshal(headerBytes, &rawHeader); err != nil {
		return nil, fmt.Errorf("parse header: %w", err)
	}

	dataStart := int64(8 + headerSize)
	tensors := make(map[string][]float32)

	for name, value := range rawHeader {
		if name == "__metadata__" {
			continue
		}
		if keep != nil && !keep(name) {
			continue
		}
		infoMap, ok := value.(map[string]any)
		if !ok {
			continue
		}
		encodedDtype, _ := infoMap["dtype"].(string)
		shapeList, _ := infoMap["shape"].([]any)
		offsetList, _ := infoMap["data_offsets"].([]any)
		if len(offsetList) != 2 {
			continue
		}
		shape := make([]int, len(shapeList))
		numElements := 1
		for i, v := range shapeList {
			shape[i] = int(v.(float64))
			numElements *= shape[i]
		}
		startOffset := int(offsetList[0].(float64))
		endOffset := int(offsetList[1].(float64))
		if endOffset < startOffset {
			return nil, fmt.Errorf("tensor %s: invalid offsets %d..%d", name, startOffset, endOffset)
		}
		byteLen := endOffset - startOffset
		tensorBytes := make([]byte, byteLen)
		if _, err := f.ReadAt(tensorBytes, dataStart+int64(startOffset)); err != nil {
			return nil, fmt.Errorf("tensor %s read: %w", name, err)
		}
		tensorData := make([]float32, numElements)
		if err := decodeTensorData(tensorBytes, encodedDtype, numElements, tensorData); err != nil {
			return nil, fmt.Errorf("tensor %s: %w", name, err)
		}
		tensors[name] = tensorData
	}
	return tensors, nil
}

// TensorNames lists tensor keys in a safetensors file (header only).
func TensorNames(filepath string) ([]string, error) {
	f, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var lenBuf [8]byte
	if _, err := f.Read(lenBuf[:]); err != nil {
		return nil, err
	}
	headerSize := binary.LittleEndian.Uint64(lenBuf[:])
	if headerSize > 256<<20 {
		return nil, fmt.Errorf("safetensors header size unreasonable: %d", headerSize)
	}
	headerBytes := make([]byte, headerSize)
	if _, err := f.Read(headerBytes); err != nil {
		return nil, err
	}
	var rawHeader map[string]any
	if err := json.Unmarshal(headerBytes, &rawHeader); err != nil {
		return nil, err
	}
	names := make([]string, 0, len(rawHeader))
	for name := range rawHeader {
		if name == "__metadata__" {
			continue
		}
		names = append(names, name)
	}
	return names, nil
}

func decodeTensorData(data []byte, dtype string, numElements int, out []float32) error {
	switch dtype {
	case "F32":
		for i := 0; i < numElements; i++ {
			offset := i * 4
			if offset+4 > len(data) {
				return fmt.Errorf("data out of bounds")
			}
			bits := binary.LittleEndian.Uint32(data[offset : offset+4])
			out[i] = math.Float32frombits(bits)
		}
	case "F16":
		for i := 0; i < numElements; i++ {
			offset := i * 2
			if offset+2 > len(data) {
				return fmt.Errorf("data out of bounds")
			}
			out[i] = float16ToFloat32(binary.LittleEndian.Uint16(data[offset : offset+2]))
		}
	case "BF16":
		for i := 0; i < numElements; i++ {
			offset := i * 2
			if offset+2 > len(data) {
				return fmt.Errorf("data out of bounds")
			}
			bf16 := binary.LittleEndian.Uint16(data[offset : offset+2])
			out[i] = math.Float32frombits(uint32(bf16) << 16)
		}
	case "F64":
		for i := 0; i < numElements; i++ {
			offset := i * 8
			if offset+8 > len(data) {
				return fmt.Errorf("data out of bounds")
			}
			bits := binary.LittleEndian.Uint64(data[offset : offset+8])
			out[i] = float32(math.Float64frombits(bits))
		}
	default:
		return fmt.Errorf("unsupported dtype: %s", dtype)
	}
	return nil
}

func float16ToFloat32(f16 uint16) float32 {
	sign := uint32((f16 >> 15) & 0x1)
	exponent := uint32((f16 >> 10) & 0x1F)
	mantissa := uint32(f16 & 0x3FF)
	var f32bits uint32
	if exponent == 0 {
		if mantissa == 0 {
			f32bits = sign << 31
		} else {
			exponent = 1
			for (mantissa & 0x400) == 0 {
				mantissa <<= 1
				exponent--
			}
			mantissa &= 0x3FF
			f32bits = (sign << 31) | ((exponent + (127 - 15)) << 23) | (mantissa << 13)
		}
	} else if exponent == 0x1F {
		f32bits = (sign << 31) | (0xFF << 23) | (mantissa << 13)
	} else {
		f32bits = (sign << 31) | ((exponent + (127 - 15)) << 23) | (mantissa << 13)
	}
	return *(*float32)(unsafe.Pointer(&f32bits))
}
