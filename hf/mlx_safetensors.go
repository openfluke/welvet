package hf

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/openfluke/welvet/quant"
)

// TensorInfo is one safetensors header entry.
type TensorInfo struct {
	Name   string
	Dtype  string
	Shape  []int
	Start  int64 // byte offset into data region
	Length int64
}

// ListTensorInfos returns header metadata (no payload).
func ListTensorInfos(filepath string) ([]TensorInfo, error) {
	f, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	header, dataStart, err := readSTHeader(f)
	if err != nil {
		return nil, err
	}
	_ = dataStart
	out := make([]TensorInfo, 0, len(header))
	for name, value := range header {
		if name == "__metadata__" {
			continue
		}
		infoMap, ok := value.(map[string]any)
		if !ok {
			continue
		}
		ti, err := parseTensorInfo(name, infoMap)
		if err != nil {
			continue
		}
		out = append(out, ti)
	}
	return out, nil
}

func readSTHeader(f *os.File) (map[string]any, int64, error) {
	var lenBuf [8]byte
	if _, err := io.ReadFull(f, lenBuf[:]); err != nil {
		return nil, 0, err
	}
	headerSize := binary.LittleEndian.Uint64(lenBuf[:])
	if headerSize > 256<<20 {
		return nil, 0, fmt.Errorf("safetensors header size unreasonable: %d", headerSize)
	}
	headerBytes := make([]byte, headerSize)
	if _, err := io.ReadFull(f, headerBytes); err != nil {
		return nil, 0, err
	}
	var rawHeader map[string]any
	if err := json.Unmarshal(headerBytes, &rawHeader); err != nil {
		return nil, 0, err
	}
	return rawHeader, int64(8 + headerSize), nil
}

func parseTensorInfo(name string, infoMap map[string]any) (TensorInfo, error) {
	dtype, _ := infoMap["dtype"].(string)
	shapeList, _ := infoMap["shape"].([]any)
	offsetList, _ := infoMap["data_offsets"].([]any)
	if len(offsetList) != 2 {
		return TensorInfo{}, fmt.Errorf("bad offsets")
	}
	shape := make([]int, len(shapeList))
	for i, v := range shapeList {
		shape[i] = int(v.(float64))
	}
	start := int64(offsetList[0].(float64))
	end := int64(offsetList[1].(float64))
	return TensorInfo{Name: name, Dtype: dtype, Shape: shape, Start: start, Length: end - start}, nil
}

// ReadTensorBytes loads raw bytes for one tensor.
func ReadTensorBytes(filepath string, ti TensorInfo) ([]byte, error) {
	f, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	_, dataStart, err := readSTHeader(f)
	if err != nil {
		return nil, err
	}
	buf := make([]byte, ti.Length)
	if _, err := f.ReadAt(buf, dataStart+ti.Start); err != nil {
		return nil, err
	}
	return buf, nil
}

// DecodeF16Tensor decodes an F16 safetensors payload to float32.
func DecodeF16Tensor(data []byte, n int) ([]float32, error) {
	if len(data) < n*2 {
		return nil, fmt.Errorf("f16 short")
	}
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = float16ToFloat32(binary.LittleEndian.Uint16(data[i*2:]))
	}
	return out, nil
}

// DecodeU32Tensor decodes a U32 safetensors payload.
func DecodeU32Tensor(data []byte, n int) ([]uint32, error) {
	if len(data) < n*4 {
		return nil, fmt.Errorf("u32 short")
	}
	out := make([]uint32, n)
	for i := 0; i < n; i++ {
		out[i] = binary.LittleEndian.Uint32(data[i*4:])
	}
	return out, nil
}

// LoadMLX1BitMatrix loads weight+scales+biases for an MLX 1-bit linear and builds a BinaryG128 blob.
// base is the tensor prefix without ".weight"/".scales"/".biases" (e.g. "...gate_proj").
func LoadMLX1BitMatrix(filepath string, index map[string]TensorInfo, base string) (*quant.Blob, error) {
	wInfo, ok := index[base+".weight"]
	if !ok {
		return nil, fmt.Errorf("missing %s.weight", base)
	}
	sInfo, ok := index[base+".scales"]
	if !ok {
		return nil, fmt.Errorf("missing %s.scales", base)
	}
	bInfo, ok := index[base+".biases"]
	if !ok {
		return nil, fmt.Errorf("missing %s.biases", base)
	}
	if wInfo.Dtype != "U32" || sInfo.Dtype != "F16" || bInfo.Dtype != "F16" {
		return nil, fmt.Errorf("%s: unexpected dtypes %s/%s/%s", base, wInfo.Dtype, sInfo.Dtype, bInfo.Dtype)
	}
	if len(wInfo.Shape) != 2 || len(sInfo.Shape) != 2 {
		return nil, fmt.Errorf("%s: bad shapes", base)
	}
	rows, words := wInfo.Shape[0], wInfo.Shape[1]
	cols := words * 32
	wBytes, err := ReadTensorBytes(filepath, wInfo)
	if err != nil {
		return nil, err
	}
	sBytes, err := ReadTensorBytes(filepath, sInfo)
	if err != nil {
		return nil, err
	}
	bBytes, err := ReadTensorBytes(filepath, bInfo)
	if err != nil {
		return nil, err
	}
	wu, err := DecodeU32Tensor(wBytes, rows*words)
	if err != nil {
		return nil, err
	}
	ns := sInfo.Shape[0] * sInfo.Shape[1]
	scales, err := DecodeF16Tensor(sBytes, ns)
	if err != nil {
		return nil, err
	}
	biases, err := DecodeF16Tensor(bBytes, ns)
	if err != nil {
		return nil, err
	}
	return quant.BlobFromMLX1Bit(wu, scales, biases, rows, cols)
}

// LoadF16Vector loads a 1-D F16 tensor as float32 (for norms, biases, A_log, …).
func LoadF16Vector(filepath string, index map[string]TensorInfo, name string) ([]float32, error) {
	ti, ok := index[name]
	if !ok {
		return nil, fmt.Errorf("missing %s", name)
	}
	if ti.Dtype != "F16" && ti.Dtype != "BF16" && ti.Dtype != "F32" {
		return nil, fmt.Errorf("%s: dtype %s", name, ti.Dtype)
	}
	n := 1
	for _, d := range ti.Shape {
		n *= d
	}
	raw, err := ReadTensorBytes(filepath, ti)
	if err != nil {
		return nil, err
	}
	switch ti.Dtype {
	case "F16":
		return DecodeF16Tensor(raw, n)
	case "F32":
		out := make([]float32, n)
		if err := decodeTensorData(raw, "F32", n, out); err != nil {
			return nil, err
		}
		return out, nil
	default:
		out := make([]float32, n)
		if err := decodeTensorData(raw, ti.Dtype, n, out); err != nil {
			return nil, err
		}
		return out, nil
	}
}

// BuildTensorIndex maps name → TensorInfo for one file.
func BuildTensorIndex(filepath string) (map[string]TensorInfo, error) {
	infos, err := ListTensorInfos(filepath)
	if err != nil {
		return nil, err
	}
	out := make(map[string]TensorInfo, len(infos))
	for _, ti := range infos {
		out[ti.Name] = ti
	}
	return out, nil
}

// SkipVisionTensor reports vision_tower / multi-modal projector keys.
func SkipVisionTensor(name string) bool {
	return strings.HasPrefix(name, "vision_tower.") ||
		strings.HasPrefix(name, "visual.") ||
		strings.Contains(name, "multi_modal_projector") ||
		strings.Contains(name, "merger.")
}
