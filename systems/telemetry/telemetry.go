// Package telemetry extracts static structural blueprints from architecture.Grid
// (loom/poly telemetry rebuild). Live UDP HUD events live in package tanhi.
//
// Tests live in github.com/openfluke/w2a — not here.
package telemetry

import (
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/systems/dna"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/cnn2"
	"github.com/openfluke/welvet/layers/cnn3"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/lstm"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/rnn"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/layers/swiglu"
)

// NetworkBlueprint contains structural information of one or more models.
type NetworkBlueprint struct {
	Models []ModelTelemetry `json:"models"`
}

// ModelTelemetry represents a single network's structure.
type ModelTelemetry struct {
	ID          string           `json:"id"`
	TotalLayers int              `json:"total_layers"`
	TotalParams int              `json:"total_parameters"`
	Layers      []LayerTelemetry `json:"layers"`
}

// LayerTelemetry contains metadata about a specific cell.
type LayerTelemetry struct {
	Z int `json:"z"`
	Y int `json:"y"`
	X int `json:"x"`
	L int `json:"l"`

	Type       string `json:"type"`
	Activation string `json:"activation,omitempty"`
	Parameters int    `json:"parameters"`

	InputShape  []int `json:"input_shape,omitempty"`
	OutputShape []int `json:"output_shape,omitempty"`

	Branches    []LayerTelemetry `json:"branches,omitempty"`
	CombineMode string           `json:"combine_mode,omitempty"`
}

// ExtractNetworkBlueprint extracts structural telemetry from a Grid.
func ExtractNetworkBlueprint(g *architecture.Grid, modelID string) ModelTelemetry {
	tel := ModelTelemetry{
		ID:     modelID,
		Layers: make([]LayerTelemetry, 0),
	}
	if g == nil {
		return tel
	}
	order := g.HopOrder()
	tel.TotalLayers = len(order)
	totalParams := 0
	for _, c := range order {
		cell := g.At(c.Z, c.Y, c.X, c.L)
		if cell == nil {
			continue
		}
		lt := ExtractLayerTelemetry(*cell)
		tel.Layers = append(tel.Layers, lt)
		totalParams += lt.Parameters
	}
	tel.TotalParams = totalParams
	return tel
}

// ExtractLayerTelemetry converts a Cell to its telemetry representation.
func ExtractLayerTelemetry(cell architecture.Cell) LayerTelemetry {
	tel := LayerTelemetry{
		Z:          cell.Layer.Z,
		Y:          cell.Layer.Y,
		X:          cell.Layer.X,
		L:          cell.Layer.L,
		Type:       cell.Layer.Type.String(),
		Activation: cell.Layer.Activation.String(),
	}

	params := paramCountFromOp(cell.Op)
	if params == 0 {
		params = paramEstimateFromMeta(cell)
	}
	tel.Parameters = params
	tel.InputShape, tel.OutputShape = shapesFromCell(cell)

	if len(cell.ParallelBranches) > 0 {
		for i := range cell.ParallelBranches {
			bt := ExtractLayerTelemetry(cell.ParallelBranches[i])
			tel.Branches = append(tel.Branches, bt)
			if params == 0 {
				tel.Parameters += bt.Parameters
			}
		}
		tel.CombineMode = cell.CombineMode
	}
	return tel
}

func paramCountFromOp(op any) int {
	n := 0
	for _, s := range dna.CollectStores(op) {
		if s != nil {
			n += s.ParamCount()
		}
	}
	return n
}

func paramEstimateFromMeta(cell architecture.Cell) int {
	l := cell.Layer
	switch l.Type {
	case core.LayerDense:
		return l.InputHeight * l.OutputHeight
	case core.LayerSoftmax:
		return 0
	case core.LayerRMSNorm:
		return l.InputHeight
	case core.LayerLayerNorm:
		return l.InputHeight * 2
	case core.LayerSwiGLU:
		return l.InputHeight * l.OutputHeight * 3
	case core.LayerEmbedding:
		return l.InputHeight * l.OutputHeight
	default:
		return 0
	}
}

func shapesFromCell(cell architecture.Cell) (in, out []int) {
	l := cell.Layer
	switch op := cell.Op.(type) {
	case *dense.Layer:
		return []int{op.Core.InputHeight}, []int{op.Core.OutputHeight}
	case *mha.Layer:
		seq := op.Cfg.MaxSeqLen
		if seq <= 0 {
			seq = 1
		}
		return []int{seq, op.Cfg.DModel}, []int{seq, op.Cfg.DModel}
	case *swiglu.Layer:
		return []int{op.Cfg.InputDim}, []int{op.Cfg.InputDim}
	case *rmsnorm.Layer:
		return []int{op.Cfg.Dim}, []int{op.Cfg.Dim}
	case *layernorm.Layer:
		return []int{op.Cfg.Dim}, []int{op.Cfg.Dim}
	case *cnn1.Layer:
		return []int{op.Cfg.InChannels, op.Cfg.SeqLen}, []int{op.Cfg.Filters, op.Cfg.OutLen()}
	case *cnn2.Layer:
		return []int{op.Cfg.InChannels, op.Cfg.Height, op.Cfg.Width}, []int{op.Cfg.Filters, op.Cfg.OutH(), op.Cfg.OutW()}
	case *cnn3.Layer:
		return []int{op.Cfg.InChannels, op.Cfg.Depth, op.Cfg.Height, op.Cfg.Width},
			[]int{op.Cfg.Filters, op.Cfg.OutD(), op.Cfg.OutH(), op.Cfg.OutW()}
	case *rnn.Layer:
		return []int{op.Cfg.SeqLen, op.Cfg.InputSize}, []int{op.Cfg.SeqLen, op.Cfg.HiddenSize}
	case *lstm.Layer:
		return []int{op.Cfg.SeqLen, op.Cfg.InputSize}, []int{op.Cfg.SeqLen, op.Cfg.HiddenSize}
	case *embedding.Layer:
		return []int{op.Cfg.VocabSize}, []int{op.Cfg.EmbeddingDim}
	case *sequential.Layer:
		return []int{op.Cfg.Dim}, []int{op.Cfg.Dim}
	case *residual.Layer:
		return []int{op.Cfg.Dim}, []int{op.Cfg.Dim}
	case *softmax.Layer:
		return []int{op.Cfg.Dim}, []int{op.Cfg.Dim}
	default:
		if l.InputHeight > 0 {
			in = []int{l.InputHeight}
		}
		if l.OutputHeight > 0 {
			out = []int{l.OutputHeight}
		}
		return in, out
	}
}
