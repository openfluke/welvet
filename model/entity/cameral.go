package entity

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/metacognition"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

// CameralSpec is the sandwich add-on in the ENTITY header (stem / hemispheres / head).
type CameralSpec struct {
	Kind      string `json:"kind"` // "stack" or "bicameral"
	TrainMode string `json:"train_mode,omitempty"`
	In        int    `json:"in,omitempty"`
	Hidden    int    `json:"hidden,omitempty"`
	Out       int    `json:"out,omitempty"`
	Root      OpNode `json:"root"`
}

// OpNode is one cameral graph node (Dense, Parallel, Stack, View, …).
type OpNode struct {
	Type        string   `json:"type"`
	Activation  string   `json:"activation,omitempty"`
	DType       string   `json:"dtype,omitempty"`
	Format      string   `json:"format,omitempty"`
	In          int      `json:"in,omitempty"`
	Out         int      `json:"out,omitempty"`
	Combine     string   `json:"combine,omitempty"`
	OutFeat     int      `json:"out_feat,omitempty"`
	SeqLen      int      `json:"seq_len,omitempty"`
	Shape       []int    `json:"shape,omitempty"`
	BranchModes []string `json:"branch_modes,omitempty"`
	ChildModes  []string `json:"child_modes,omitempty"`
	AltTimes    int      `json:"alt_times,omitempty"`
	InChannels  int      `json:"in_channels,omitempty"`
	Filters     int      `json:"filters,omitempty"`
	Kernel      int      `json:"kernel,omitempty"`
	Stride      int      `json:"stride,omitempty"`
	Padding     int      `json:"padding,omitempty"`
	Children    []OpNode `json:"children,omitempty"`
	Branches    []OpNode `json:"branches,omitempty"`
	Weight      string   `json:"weight,omitempty"`
	Gate        string   `json:"gate,omitempty"`
	Observed    string   `json:"observed,omitempty"`
	Proj        string   `json:"proj,omitempty"`
}

type blobSink struct {
	payload []byte
	blobs   []WeightBlob
}

func (s *blobSink) addStore(path string, st *weights.Store) error {
	if st == nil {
		return fmt.Errorf("entity: nil store for %q", path)
	}
	snap, err := weights.TakeSnapshot(st)
	if err != nil {
		return fmt.Errorf("entity store %q: %w", path, err)
	}
	off := len(s.payload)
	s.payload = append(s.payload, snap.Raw...)
	s.blobs = append(s.blobs, WeightBlob{
		Path:   path,
		Offset: uint64(off),
		Length: uint64(len(snap.Raw)),
		DType:  snap.DType.String(),
		Format: snap.Format.String(),
		Rows:   snap.Rows,
		Cols:   snap.Cols,
		Scale:  snap.Scale,
		Native: true,
	})
	if len(snap.Bias) > 0 {
		raw := weights.EncodeF64LE(snap.Bias)
		boff := len(s.payload)
		s.payload = append(s.payload, raw...)
		s.blobs = append(s.blobs, WeightBlob{
			Path:   path + ".bias",
			Offset: uint64(boff),
			Length: uint64(len(raw)),
			DType:  "float64",
			Format: "none",
			Native: true,
		})
	}
	return nil
}

func restoreStore(ef *File, path string) (*weights.Store, error) {
	blob, err := ef.findBlob(path)
	if err != nil {
		return nil, err
	}
	raw, err := ef.LoadBlobBytes(path)
	if err != nil {
		return nil, err
	}
	var bias []float64
	if b, berr := ef.findBlob(path + ".bias"); berr == nil && b != nil {
		br, err := ef.LoadBlobBytes(path + ".bias")
		if err != nil {
			return nil, err
		}
		bias, err = weights.DecodeF64LE(br)
		if err != nil {
			return nil, err
		}
	}
	return weights.Restore(weights.Snapshot{
		DType:  core.ParseDType(blob.DType),
		Format: quant.ParseFormatName(blob.Format),
		Rows:   blob.Rows,
		Cols:   blob.Cols,
		Scale:  blob.Scale,
		Raw:    raw,
		Bias:   bias,
	})
}

func denseFromStore(in, out int, act core.ActivationType, s *weights.Store) (*dense.Layer, error) {
	if s == nil {
		return nil, fmt.Errorf("entity: nil dense store")
	}
	if s.Cols > 0 && s.Rows > 0 {
		in, out = s.Cols, s.Rows
	}
	l, err := dense.New(in, out, act, s.DType)
	if err != nil {
		return nil, err
	}
	l.Weights = s
	l.Core.DType = s.DType
	return l, nil
}

func modesToStrings(ms []parallel.TrainMode) []string {
	if len(ms) == 0 {
		return nil
	}
	out := make([]string, len(ms))
	for i, m := range ms {
		out[i] = m.String()
	}
	return out
}

func stringsToModes(ss []string) ([]parallel.TrainMode, error) {
	if len(ss) == 0 {
		return nil, nil
	}
	out := make([]parallel.TrainMode, len(ss))
	for i, s := range ss {
		m, err := parallel.ParseTrainMode(s)
		if err != nil {
			return nil, err
		}
		out[i] = m
	}
	return out, nil
}

func encodeOp(op any, path string, sink *blobSink) (OpNode, error) {
	if op == nil {
		return OpNode{}, fmt.Errorf("entity: nil cameral op at %s", path)
	}
	switch v := op.(type) {
	case *dense.Layer:
		n := OpNode{
			Type:       "Dense",
			Activation: v.Core.Activation.String(),
			DType:      v.Core.DType.String(),
			In:         v.Core.InputHeight,
			Out:        v.Core.OutputHeight,
			Weight:     path + ".w",
		}
		if v.Weights != nil {
			n.Format = v.Weights.Format.String()
			if err := sink.addStore(n.Weight, v.Weights); err != nil {
				return n, err
			}
		}
		return n, nil
	case *parallel.View:
		return OpNode{Type: "View", Shape: append([]int(nil), v.Shape...)}, nil
	case *parallel.Stack:
		n := OpNode{
			Type:       "Stack",
			ChildModes: modesToStrings(v.ChildModes),
			AltTimes:   v.AltTimes,
			In:         v.Core.InputHeight,
			Out:        v.Core.OutputHeight,
			Children:   make([]OpNode, len(v.Children)),
		}
		for i, ch := range v.Children {
			cn, err := encodeOp(ch, fmt.Sprintf("%s.c%d", path, i), sink)
			if err != nil {
				return n, err
			}
			n.Children[i] = cn
		}
		return n, nil
	case *parallel.Layer:
		n := OpNode{
			Type:        "Parallel",
			DType:       v.Core.DType.String(),
			In:          v.Cfg.Dim,
			Out:         v.Core.OutputHeight,
			OutFeat:     v.Cfg.OutFeat,
			Combine:     string(v.Cfg.Combine),
			SeqLen:      v.Cfg.SeqLen,
			BranchModes: modesToStrings(v.BranchModes),
			AltTimes:    v.AltTimes,
			Branches:    make([]OpNode, len(v.Branches)),
		}
		for i, b := range v.Branches {
			bn, err := encodeOp(b, fmt.Sprintf("%s.b%d", path, i), sink)
			if err != nil {
				return n, err
			}
			n.Branches[i] = bn
		}
		if v.Gate != nil {
			n.Gate = path + ".gate"
			if err := sink.addStore(n.Gate, v.Gate.Weights); err != nil {
				return n, err
			}
		}
		return n, nil
	case *metacognition.Layer:
		n := OpNode{
			Type:     "Metacognition",
			DType:    v.Core.DType.String(),
			In:       v.Cfg.Dim,
			Out:      v.Cfg.Dim,
			SeqLen:   v.Cfg.SeqLen,
			Observed: path + ".observed",
		}
		if v.Observed != nil && v.Observed.Weights != nil {
			n.Format = v.Observed.Weights.Format.String()
			if err := sink.addStore(n.Observed, v.Observed.Weights); err != nil {
				return n, err
			}
		}
		return n, nil
	case *cnn1.Layer:
		n := OpNode{
			Type:       "CNN1",
			Activation: v.Cfg.Activation.String(),
			DType:      v.Core.DType.String(),
			InChannels: v.Cfg.InChannels,
			Filters:    v.Cfg.Filters,
			SeqLen:     v.Cfg.SeqLen,
			Kernel:     v.Cfg.Kernel,
			Stride:     v.Cfg.Stride,
			Padding:    v.Cfg.Padding,
			Proj:       path + ".proj",
		}
		if v.Proj != nil && v.Proj.Weights != nil {
			n.Format = v.Proj.Weights.Format.String()
			if err := sink.addStore(n.Proj, v.Proj.Weights); err != nil {
				return n, err
			}
		}
		return n, nil
	default:
		return OpNode{}, fmt.Errorf("entity: unsupported cameral op %T at %s", op, path)
	}
}

func decodeOp(ef *File, n OpNode) (any, error) {
	switch n.Type {
	case "Dense":
		s, err := restoreStore(ef, n.Weight)
		if err != nil {
			return nil, err
		}
		return denseFromStore(n.In, n.Out, core.ParseActivation(n.Activation), s)
	case "View":
		return parallel.NewView(n.Shape...)
	case "Stack":
		children := make([]any, len(n.Children))
		for i := range n.Children {
			ch, err := decodeOp(ef, n.Children[i])
			if err != nil {
				return nil, fmt.Errorf("stack child %d: %w", i, err)
			}
			children[i] = ch
		}
		s, err := parallel.NewStack(children...)
		if err != nil {
			return nil, err
		}
		modes, err := stringsToModes(n.ChildModes)
		if err != nil {
			return nil, err
		}
		s.SetChildModes(modes...)
		s.AltTimes = n.AltTimes
		return s, nil
	case "Parallel":
		branches := make([]any, len(n.Branches))
		for i := range n.Branches {
			b, err := decodeOp(ef, n.Branches[i])
			if err != nil {
				return nil, fmt.Errorf("parallel branch %d: %w", i, err)
			}
			branches[i] = b
		}
		var gate *dense.Layer
		if n.Gate != "" {
			gs, err := restoreStore(ef, n.Gate)
			if err != nil {
				return nil, err
			}
			g, err := denseFromStore(n.In, len(branches), core.ActivationLinear, gs)
			if err != nil {
				return nil, err
			}
			gate = g
		}
		cfg := parallel.Config{
			Dim: n.In, OutFeat: n.OutFeat, Branches: len(branches),
			Combine: parallel.CombineMode(n.Combine), SeqLen: n.SeqLen,
		}
		l, err := parallel.NewFromBranches(cfg, branches, gate)
		if err != nil {
			return nil, err
		}
		modes, err := stringsToModes(n.BranchModes)
		if err != nil {
			return nil, err
		}
		l.SetBranchModes(modes...)
		l.AltTimes = n.AltTimes
		if n.Out > 0 {
			l.Core.OutputHeight = n.Out
		}
		return l, nil
	case "Metacognition":
		l, err := metacognition.New(metacognition.Config{Dim: n.In, SeqLen: n.SeqLen})
		if err != nil {
			return nil, err
		}
		if n.Observed != "" {
			s, err := restoreStore(ef, n.Observed)
			if err != nil {
				return nil, err
			}
			d, err := denseFromStore(n.In, n.In, core.ActivationLinear, s)
			if err != nil {
				return nil, err
			}
			l.Observed = d
			l.Core.DType = s.DType
		}
		return l, nil
	case "CNN1":
		cfg := cnn1.Config{
			InChannels: n.InChannels, Filters: n.Filters, SeqLen: n.SeqLen,
			Kernel: n.Kernel, Stride: n.Stride, Padding: n.Padding,
			Activation: core.ParseActivation(n.Activation),
		}
		l, err := cnn1.New(cfg)
		if err != nil {
			return nil, err
		}
		if n.Proj != "" {
			s, err := restoreStore(ef, n.Proj)
			if err != nil {
				return nil, err
			}
			d, err := denseFromStore(s.Cols, s.Rows, cfg.Activation, s)
			if err != nil {
				return nil, err
			}
			l.Proj = d
			l.Core.DType = s.DType
		}
		return l, nil
	default:
		return nil, fmt.Errorf("entity: unsupported cameral type %q", n.Type)
	}
}

func inferCameralKind(s *parallel.Stack) (kind string, in, hidden, out int) {
	kind = "stack"
	if s == nil || len(s.Children) != 3 {
		if s != nil {
			in, out = s.Core.InputHeight, s.Core.OutputHeight
		}
		return
	}
	stem, ok0 := s.Children[0].(*dense.Layer)
	hemi, ok1 := s.Children[1].(*parallel.Layer)
	head, ok2 := s.Children[2].(*dense.Layer)
	if !ok0 || !ok1 || !ok2 || hemi.Cfg.Combine != parallel.CombineAdd {
		in, out = s.Core.InputHeight, s.Core.OutputHeight
		return
	}
	kind = "bicameral"
	in = stem.Core.InputHeight
	hidden = stem.Core.OutputHeight
	out = head.Core.OutputHeight
	return
}

// WriteCameralFile writes a sandwich Stack as a .entity checkpoint.
// parentMode is stored in the header (empty / inherit when ModeInherit).
func WriteCameralFile(outPath string, stack *parallel.Stack, parentMode parallel.TrainMode) error {
	if stack == nil {
		return fmt.Errorf("entity.WriteCameralFile: nil stack")
	}
	sink := &blobSink{}
	root, err := encodeOp(stack, "cameral", sink)
	if err != nil {
		return err
	}
	kind, in, hidden, out := inferCameralKind(stack)
	spec := &CameralSpec{
		Kind: kind,
		In:   in, Hidden: hidden, Out: out,
		Root: root,
	}
	if parentMode != parallel.ModeInherit {
		spec.TrainMode = parentMode.String()
	}
	doc := headerDoc{
		FormatVersion: FormatVersion,
		Engine:        "welvet",
		Status:        "packed",
		Cameral:       spec,
		Blobs:         sink.blobs,
	}
	return writeEntityFile(outPath, doc, sink.payload)
}

// LoadCameral reconstructs a sandwich Stack from a cameral .entity.
func LoadCameral(path string) (*parallel.Stack, parallel.TrainMode, error) {
	ef, err := Open(path)
	if err != nil {
		return nil, parallel.ModeInherit, err
	}
	defer ef.Close()
	return ef.LoadCameral()
}

// LoadCameral rebuilds the sandwich from an open cameral entity.
func (ef *File) LoadCameral() (*parallel.Stack, parallel.TrainMode, error) {
	if ef == nil || ef.hdr == nil || ef.hdr.Cameral == nil {
		return nil, parallel.ModeInherit, fmt.Errorf("entity: not a cameral checkpoint")
	}
	spec := ef.hdr.Cameral
	op, err := decodeOp(ef, spec.Root)
	if err != nil {
		return nil, parallel.ModeInherit, err
	}
	stack, ok := op.(*parallel.Stack)
	if !ok {
		return nil, parallel.ModeInherit, fmt.Errorf("entity: cameral root is %T, want Stack", op)
	}
	mode, err := parallel.ParseTrainMode(spec.TrainMode)
	if err != nil {
		return nil, parallel.ModeInherit, err
	}
	return stack, mode, nil
}
