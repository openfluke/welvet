package nativemetal

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/cnn2"
	"github.com/openfluke/welvet/layers/cnn3"
	"github.com/openfluke/welvet/layers/convt1"
	"github.com/openfluke/welvet/layers/convt2"
	"github.com/openfluke/welvet/layers/convt3"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/layers/gdn"
	"github.com/openfluke/welvet/layers/kmeans"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/lstm"
	"github.com/openfluke/welvet/layers/mamba"
	"github.com/openfluke/welvet/layers/metacognition"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/rnn"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/systems/dna"
	"github.com/openfluke/welvet/weights"
	"reflect"
)

type disabledNode struct{}

type encoder struct {
	bytes.Buffer
	err   error
	nodes int
}

func (e *encoder) put(v any) {
	if e.err == nil {
		e.err = binary.Write(&e.Buffer, binary.LittleEndian, v)
	}
}
func (e *encoder) ints(v ...int) {
	for _, x := range v {
		if x < -(1<<31) || x > (1<<31)-1 {
			e.err = fmt.Errorf("nativemetal: descriptor integer outside supported range")
			return
		}
		e.put(int32(x))
	}
}
func (e *encoder) vec(v any, n int) { e.ints(n); e.put(v) }
func (e *encoder) zeros(n int) {
	if n <= 0 || n > 1<<26 {
		e.err = fmt.Errorf("nativemetal: invalid weight count %d", n)
		return
	}
	e.put(make([]float32, n))
}
func (e *encoder) fail(text string) {
	if e.err == nil {
		e.err = fmt.Errorf("nativemetal: %s", text)
	}
}
func nilOp(op any) bool {
	return op == nil || (reflect.ValueOf(op).Kind() == reflect.Ptr && reflect.ValueOf(op).IsNil())
}
func nodeStores(op any) []*weights.Store { return dna.CollectStores(op) }
func dtype(op any) int {
	ss := nodeStores(op)
	if len(ss) > 0 && ss[0] != nil {
		return int(ss[0].DType)
	}
	return int(core.DTypeFloat32)
}
func children(op any) []any {
	switch l := op.(type) {
	case *parallel.Stack:
		return l.Children
	case *parallel.Layer:
		return l.Branches
	}
	return nil
}
func denseChildren(e *encoder, ops []any, dim int) int {
	act := -999
	for _, op := range ops {
		d, ok := op.(*dense.Layer)
		if !ok || d == nil || d.Weights.Rows != dim || d.Weights.Cols != dim {
			e.fail("Sequential/Residual mount requires square Dense children; use parallel.Stack for heterogeneous chains")
			return -1
		}
		if act == -999 {
			act = int(d.Core.Activation)
		} else if act != int(d.Core.Activation) {
			e.fail("Sequential/Residual children require a shared activation")
		}
	}
	return act
}
func (e *encoder) node(op any, depth int) {
	if e.err != nil {
		return
	}
	e.nodes++
	if depth > 64 || e.nodes > 4096 || nilOp(op) {
		e.fail("invalid, cyclic or oversized model")
		return
	}
	dt := int(core.DTypeFloat32)
	switch op.(type) {
	case *parallel.Stack, *parallel.Layer:
	default:
		dt = dtype(op)
	}
	switch l := op.(type) {
	case disabledNode:
		e.ints(0)
	case *dense.Layer:
		e.ints(1, l.Weights.Cols, l.Weights.Rows, int(l.Core.Activation), dt)
		e.zeros(l.Weights.Rows * l.Weights.Cols)
		e.put(uint32(0))
	case *rmsnorm.Layer:
		e.ints(2, l.Cfg.Dim, dt)
		e.put(l.Cfg.Eps)
		e.zeros(l.Cfg.Dim)
	case *layernorm.Layer:
		e.ints(3, l.Cfg.Dim, dt)
		e.put(l.Cfg.Eps)
		e.zeros(l.Cfg.Dim * 2)
	case *swiglu.Layer:
		e.ints(4, l.Cfg.InputDim, l.Cfg.IntermediateDim, dt)
		e.zeros(l.Cfg.InputDim * l.Cfg.IntermediateDim * 3)
	case *softmax.Layer:
		c := l.Cfg
		if c.Kind == softmax.KindGumbel || len(c.Mask) > 0 || len(c.HierarchyLevels) > 0 || c.Rows != 0 || (c.EntmaxAlpha != 0 && c.EntmaxAlpha != 1.5) {
			e.fail("this Softmax mask/grid/hierarchy/Gumbel configuration has no mounted descriptor yet")
			return
		}
		e.ints(5, c.Dim, dt, int(c.Kind))
		e.put(c.Temperature)
	case *embedding.Layer:
		e.ints(6, l.Cfg.VocabSize, l.Cfg.EmbeddingDim, l.Cfg.SeqLen, int(l.Core.Activation), dt)
		e.zeros(l.Cfg.VocabSize * l.Cfg.EmbeddingDim)
	case *mha.Layer:
		c := l.Cfg
		if l.KVOffset != 0 || l.CustomAllow != nil || len(l.QNormWeight) > 0 || len(l.KNormWeight) > 0 || c.Mode != 0 || c.Softmax != 0 || c.QKNorm || c.Dropout != 0 || c.ScaleOverride != 0 || c.Window != 0 || c.PrefixLen != 0 || len(c.ALiBiSlopes) > 0 || (c.RoPETheta != 0 && c.RoPETheta != 10000) {
			e.fail("this advanced MHA configuration has no mounted descriptor yet")
			return
		}
		e.ints(7, c.MaxSeqLen, c.DModel, c.NumHeads, c.NumKVHeads, c.HeadDim, c.MaxSeqLen, int(c.Mask), int(c.Pos), dt)
		e.zeros(l.Q.Weights.Rows*l.Q.Weights.Cols + l.K.Weights.Rows*l.K.Weights.Cols + l.V.Weights.Rows*l.V.Weights.Cols + l.O.Weights.Rows*l.O.Weights.Cols)
	case *cnn1.Layer:
		c := l.Cfg
		e.ints(8, c.InChannels, c.Filters, c.SeqLen, c.Kernel, c.Stride, c.Padding, int(c.Activation), dt)
		e.zeros(c.Filters * c.InChannels * c.Kernel)
	case *cnn2.Layer:
		c := l.Cfg
		e.ints(9, c.InChannels, c.Filters, c.Height, c.Width, c.Kernel, c.Stride, c.Padding, int(c.Activation), dt)
		e.zeros(c.Filters * c.InChannels * c.Kernel * c.Kernel)
	case *cnn3.Layer:
		c := l.Cfg
		e.ints(10, c.InChannels, c.Filters, c.Depth, c.Height, c.Width, c.Kernel, c.Stride, c.Padding, int(c.Activation), dt)
		e.zeros(c.Filters * c.InChannels * c.Kernel * c.Kernel * c.Kernel)
	case *convt1.Layer:
		c := l.Cfg
		e.ints(11, c.InChannels, c.Filters, c.SeqLen, c.Kernel, c.Stride, c.Padding, c.OutputPadding, int(c.Activation), dt)
		e.zeros(c.Filters * c.InChannels * c.Kernel)
	case *convt2.Layer:
		c := l.Cfg
		e.ints(12, c.InChannels, c.Filters, c.Height, c.Width, c.Kernel, c.Stride, c.Padding, c.OutputPadding, int(c.Activation), dt)
		e.zeros(c.Filters * c.InChannels * c.Kernel * c.Kernel)
	case *convt3.Layer:
		c := l.Cfg
		e.ints(13, c.InChannels, c.Filters, c.Depth, c.Height, c.Width, c.Kernel, c.Stride, c.Padding, c.OutputPadding, int(c.Activation), dt)
		e.zeros(c.Filters * c.InChannels * c.Kernel * c.Kernel * c.Kernel)
	case *rnn.Layer:
		c := l.Cfg
		e.ints(14, c.InputSize, c.HiddenSize, c.SeqLen, dt)
		e.zeros(c.WeightCount())
	case *lstm.Layer:
		c := l.Cfg
		e.ints(15, c.InputSize, c.HiddenSize, c.SeqLen, dt)
		e.zeros(c.WeightCount())
	case *sequential.Layer:
		c := l.Cfg
		ops := l.ChildOps()
		act := denseChildren(e, ops, c.Dim)
		e.ints(16, c.Dim, c.SeqLen, len(ops), act, dt)
		e.zeros(len(ops) * c.Dim * c.Dim)
	case *residual.Layer:
		c := l.Cfg
		ops := l.ChildOps()
		act := denseChildren(e, ops, c.Dim)
		e.ints(17, c.Dim, c.SeqLen, len(ops), act, dt)
		e.zeros(len(ops) * c.Dim * c.Dim)
	case *mamba.Layer:
		c := l.Cfg
		e.ints(18, c.DModel, c.DState, c.Expand, c.SeqLen, dt)
		e.zeros(l.InProj.Weights.Rows*l.InProj.Weights.Cols + l.OutProj.Weights.Rows*l.OutProj.Weights.Cols)
		e.put(l.ALog)
		e.put(l.DSkip)
	case *parallel.Stack:
		e.ints(19, len(l.Children))
		for _, op := range l.Children {
			e.node(op, depth+1)
		}
	case *metacognition.Layer:
		if !reflect.DeepEqual(l.Rules, metacognition.DefaultStabilityRules()) {
			e.fail("mount Metacognition with fresh default rules; custom/live rule state needs a separate descriptor")
			return
		}
		e.ints(20, l.Cfg.Dim, l.Cfg.SeqLen, dt)
		e.zeros(l.Cfg.Dim * l.Cfg.Dim)
	case *gdn.Layer:
		c := l.Cfg
		if l.HasState || c.Eps != 0 && c.Eps != 1e-6 {
			e.fail("GDN mount requires reset state and default epsilon")
			return
		}
		e.ints(21, c.HiddenSize, 1, c.NumKeyHeads, c.NumValueHeads, c.KeyHeadDim, c.ValueHeadDim, c.ConvKernel, int(core.DTypeFloat32))
		for _, b := range []*quant.Blob{l.InQKV, l.InZ, l.InB, l.InA, l.Out} {
			if b == nil || b.Format != quant.FormatNone {
				e.fail("GDN mount supports Float32 FormatNone projections")
				return
			}
			v, err := quant.Unpack(b)
			if err != nil {
				e.err = err
				return
			}
			e.put(v)
		}
		e.put(l.ConvWeight)
		e.put(l.ALog)
		e.put(l.DtBias)
		e.put(l.NormGamma)
	case *kmeans.Layer:
		c := l.Cfg
		e.ints(22, c.NumClusters, c.FeatureDim, 1, outputMode(c.OutputMode), int(c.Activation))
		e.put(c.Temperature)
		e.ints(dt)
		e.zeros(c.NumClusters * c.FeatureDim)
	case *parallel.Layer:
		c := l.Cfg
		if l.Rotate != nil || l.CamKit != nil && (l.CamKit.DNAReg != 0 || l.CamKit.Dream != nil) {
			e.fail("rotation, DNA regularization and Dream replay require separate mount contracts")
			return
		}
		if l.AltTimes > 1 {
			e.fail("mounted Parallel currently supports AltTimes=1")
			return
		}
		e.ints(23, combineMode(c.Combine), c.Dim, c.OutFeat, c.SeqLen, c.SparseK)
		e.put(c.DisagreeBeta)
		e.ints(len(l.Branches))
		for _, op := range l.Branches {
			e.node(op, depth+1)
		}
		if l.Gate == nil {
			e.ints(0)
		} else {
			e.ints(1)
			e.node(l.Gate, depth+1)
		}
	default:
		e.fail(fmt.Sprintf("unsupported mounted operation %T", op))
	}
}
func (e *encoder) controls(op any) {
	var modes []parallel.TrainMode
	var rates []float64
	var coef, threshold, last float64
	var sync *parallel.CamSyncConfig
	switch l := op.(type) {
	case *parallel.Stack:
		modes = l.ChildModes
		sync = l.CamSync
		if l.AltTimes > 1 {
			e.fail("mounted Stack currently supports AltTimes=1")
		}
	case *parallel.Layer:
		modes = l.BranchModes
		rates = l.BranchLRs
		sync = l.CamSync
		if l.CamKit != nil {
			coef = l.CamKit.ShadowCoef
			threshold = l.CamKit.SurpriseThresh
			last = l.CamKit.LastLoss
		}
	}
	e.ints(len(modes))
	for _, m := range modes {
		e.ints(int(m))
	}
	e.vec(rates, len(rates))
	e.put(coef)
	e.put(threshold)
	e.put(last)
	if sync == nil {
		e.ints(0)
		return
	}
	e.ints(1)
	if sync.Enabled {
		e.ints(1)
	} else {
		e.ints(0)
	}
	e.put(sync.Alpha)
	e.ints(int(sync.When))
	e.vec(sync.BranchAlpha, len(sync.BranchAlpha))
	e.ints(len(sync.Groups))
	for _, g := range sync.Groups {
		e.ints(len(g))
		e.ints(g...)
	}
	e.ints(len(sync.Cross))
	for _, p := range sync.Cross {
		e.ints(p.A.StackIdx, p.A.Branch, p.A.Store, p.B.StackIdx, p.B.Branch, p.B.Store)
	}
}
func (e *encoder) tree(op any) {
	e.controls(op)
	for _, c := range children(op) {
		e.tree(c)
	}
}
func (e *encoder) store(s *weights.Store) {
	e.ints(int(s.DType), s.Rows, s.Cols)
	e.put(s.Scale)
	var f []float32
	if s.DType == core.DTypeFloat32 {
		var err error
		f, err = weights.MatrixF32(s)
		if err != nil {
			e.err = err
			return
		}
	}
	e.vec(f, len(f))
	e.vec(s.Native, len(s.Native))
	e.vec(s.Bias, len(s.Bias))
}

// A mounted model owns snapshots of its source stores. Shared mutable stores are
// rejected until native tied-parameter update semantics have their own gate.
type binding struct {
	stores []*weights.Store
	extras []any
}

func collectExtra(op any, b *binding) {
	switch op.(type) {
	case *mamba.Layer, *gdn.Layer:
		b.extras = append(b.extras, op)
	}
	for _, c := range children(op) {
		collectExtra(c, b)
	}
}
func encodeModel(root any) (payload []byte, bound binding, err error) {
	defer func() {
		if v := recover(); v != nil {
			payload = nil
			bound = binding{}
			err = fmt.Errorf("nativemetal: invalid model: %v", v)
		}
	}()
	return encodeModelChecked(root)
}
func encodeModelChecked(root any) ([]byte, binding, error) {
	var e encoder
	var b binding
	var ops []any
	kind := 0
	dims := [4]int{1, 1, 1, 1}
	var rootControls any
	switch r := root.(type) {
	case *parallel.Stack:
		if r == nil {
			return nil, b, fmt.Errorf("nativemetal: nil Stack")
		}
		ops = r.Children
		rootControls = r
	case *parallel.Layer:
		ops = []any{r}
		kind = 1
	case *architecture.Grid:
		if r == nil {
			return nil, b, fmt.Errorf("nativemetal: nil Grid")
		}
		kind = 2
		dims = [4]int{r.Depth, r.Rows, r.Cols, r.LayersPerCell}
		for _, c := range r.Cells {
			if nilOp(c.Op) && c.Layer.IsDisabled {
				ops = append(ops, disabledNode{})
			} else {
				ops = append(ops, c.Op)
			}
		}
	default:
		ops = []any{root}
	}
	e.put(uint32(0x314e4d57))
	e.ints(kind, len(ops))
	e.ints(dims[:]...)
	for _, op := range ops {
		e.node(op, 0)
	}
	if e.err != nil {
		return nil, b, e.err
	}
	e.controls(rootControls)
	for _, op := range ops {
		e.tree(op)
		b.stores = append(b.stores, nodeStores(op)...)
		collectExtra(op, &b)
	}
	seen := map[*weights.Store]bool{}
	for _, s := range b.stores {
		if s == nil || s.Format != quant.FormatNone || seen[s] {
			return nil, b, fmt.Errorf("nativemetal: nil, packed or shared weight store requires a separate mount contract")
		}
		seen[s] = true
	}
	e.ints(len(b.stores))
	for _, s := range b.stores {
		e.store(s)
	}
	if kind == 2 {
		g := root.(*architecture.Grid)
		for _, c := range g.Cells {
			en := 1
			if c.Layer.IsDisabled {
				en = 0
			}
			remote := 0
			if c.IsRemoteLink {
				remote = 1
			}
			e.ints(en, remote, c.TargetZ, c.TargetY, c.TargetX, c.TargetL)
		}
	}
	return e.Bytes(), b, e.err
}

func outputMode(m kmeans.OutputMode) int {
	if m == kmeans.OutputFeatures {
		return 1
	}
	return 0
}
func combineMode(m parallel.CombineMode) int {
	switch m {
	case parallel.CombineConcat:
		return 0
	case parallel.CombineAdd:
		return 1
	case parallel.CombineAvg:
		return 2
	case parallel.CombineFilter:
		return 3
	case parallel.CombineMax:
		return 4
	case parallel.CombineSparseK:
		return 5
	case parallel.CombineDisagree:
		return 6
	}
	return -1
}
