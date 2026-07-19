package transformer

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/entity"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

func specPackFormat(spec *entity.TransformerSpec) quant.Format {
	if spec == nil || spec.PackFormat == "" {
		return quant.FormatNone
	}
	return quant.ParseFormatName(spec.PackFormat)
}

func loadWeightStore(ef *entity.File, path string, rows, cols int, packFmt quant.Format) (*weights.Store, error) {
	if packFmt != quant.FormatNone {
		b, err := ef.LoadQuantBlob(path)
		if err != nil {
			return nil, err
		}
		return weights.FromBlob(b)
	}
	data, err := ef.LoadBlob(path)
	if err != nil {
		return nil, err
	}
	return weights.New(rows, cols, data, core.DTypeFloat32, quant.FormatNone)
}

func denseFromStore(in, out int, act core.ActivationType, ws *weights.Store) *dense.Layer {
	return &dense.Layer{
		Core: core.Layer{
			Type:         core.LayerDense,
			DType:        core.DTypeFloat32,
			Activation:   act,
			InputHeight:  in,
			OutputHeight: out,
			TileSize:     32,
			MultiCore:    true,
		},
		Weights: ws,
		Exec: core.ExecConfig{
			Backend:   core.BackendCPUTiled,
			MultiCore: true,
			TileSize:  32,
		},
	}
}

func loadLMHead(ef *entity.File, m *Model, packFmt quant.Format) error {
	if packFmt != quant.FormatNone {
		if m.LMHeadTied {
			b, err := ef.LoadQuantBlob("transformer.lm_head.packed")
			if err != nil {
				return err
			}
			m.lmHeadPacked = b
			return nil
		}
		ws, err := loadWeightStore(ef, "transformer.lm_head", m.VocabSize, m.HiddenSize, packFmt)
		if err != nil {
			return err
		}
		m.lmHead = ws
		return nil
	}
	if m.LMHeadTied {
		return nil
	}
	lm, err := ef.LoadBlob("transformer.lm_head")
	if err != nil {
		return err
	}
	ws, err := weights.New(m.VocabSize, m.HiddenSize, lm, core.DTypeFloat32, quant.FormatNone)
	if err != nil {
		return fmt.Errorf("lm_head: %w", err)
	}
	m.lmHead = ws
	return nil
}
