package nativemetal

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"github.com/openfluke/welvet/core"
	"math"
)

func decodeSegmentTensor(r *bytes.Reader) (*core.Tensor[float32], error) {
	var rank int32
	if err := binary.Read(r, binary.LittleEndian, &rank); err != nil || rank < 1 || rank > 5 {
		return nil, fmt.Errorf("nativemetal: invalid segment rank")
	}
	shape := make([]int, rank)
	n := 1
	for i := range shape {
		var d int32
		if err := binary.Read(r, binary.LittleEndian, &d); err != nil || d < 0 || d > 1<<26 || (d > 0 && n > (1<<26)/int(d)) {
			return nil, fmt.Errorf("nativemetal: invalid segment shape")
		}
		shape[i] = int(d)
		n *= int(d)
	}
	data, err := readVector[float32](r)
	if err != nil || len(data) != n {
		return nil, fmt.Errorf("nativemetal: invalid segment data")
	}
	return &core.Tensor[float32]{Shape: shape, Data: data}, nil
}
func (s *Session) segmentPair(kind int, x, gy *core.Tensor[float32]) (*core.Tensor[float32], *core.Tensor[float32], error) {
	if s == nil {
		return nil, nil, fmt.Errorf("nativemetal: nil session")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ready(); err != nil {
		return nil, nil, err
	}
	var e encoder
	encodeTensor(&e, x)
	if kind == 1 {
		encodeTensor(&e, gy)
	}
	if e.err != nil {
		return nil, nil, e.err
	}
	data, err := s.d.segment(kind, 0, e.Bytes())
	if err != nil {
		return nil, nil, err
	}
	r := bytes.NewReader(data)
	a, err := decodeSegmentTensor(r)
	if err != nil {
		return nil, nil, err
	}
	b, err := decodeSegmentTensor(r)
	if err != nil {
		return nil, nil, err
	}
	if r.Len() != 0 {
		return nil, nil, fmt.Errorf("nativemetal: trailing segment result")
	}
	return a, b, nil
}

// ForwardPair returns native pre-activation and output for a Stack/Parallel segment.
func (s *Session) ForwardPair(x *core.Tensor[float32]) (*core.Tensor[float32], *core.Tensor[float32], error) {
	return s.segmentPair(0, x, nil)
}

// Backward accepts the global loss gradient at this segment's output. Native
// Metal recomputes its local tape and returns dInput/dWeights without updating.
func (s *Session) Backward(x, gradOut *core.Tensor[float32]) (*core.Tensor[float32], *core.Tensor[float32], error) {
	return s.segmentPair(1, x, gradOut)
}

// ApplySGD updates native segment parameters. Explicit sessions still require
// SyncToGo before Go-side execution or saving.
func (s *Session) ApplySGD(dw *core.Tensor[float32], lr float64) error {
	if s == nil {
		return fmt.Errorf("nativemetal: nil session")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ready(); err != nil {
		return err
	}
	if math.IsNaN(lr) || math.IsInf(lr, 0) {
		return fmt.Errorf("nativemetal: invalid rate")
	}
	if dw == nil || len(dw.Data) == 0 {
		return nil
	}
	var e encoder
	encodeTensor(&e, dw)
	if e.err != nil {
		return e.err
	}
	_, err := s.d.segment(2, lr, e.Bytes())
	return err
}

func (s *Session) denseProxy(x, gy, pre *core.Tensor[float32], kind int) (*core.Tensor[float32], error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ready(); err != nil {
		return nil, err
	}
	var e encoder
	encodeTensor(&e, x)
	encodeTensor(&e, gy)
	if pre == nil {
		e.ints(0)
	} else {
		e.ints(1)
		encodeTensor(&e, pre)
	}
	if e.err != nil {
		return nil, e.err
	}
	data, err := s.d.segment(kind+3, 0, e.Bytes())
	if err != nil {
		return nil, err
	}
	r := bytes.NewReader(data)
	out, err := decodeSegmentTensor(r)
	if err == nil && r.Len() != 0 {
		err = fmt.Errorf("nativemetal: trailing proxy output")
	}
	return out, err
}
