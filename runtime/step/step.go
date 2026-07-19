// Package step implements the discrete-time volumetric step mesh (loom/poly step.go).
//
// Unlike sequential forward.Forward (one full chain per call), each StepForward tick
// updates every grid cell simultaneously from double-buffered activations. Remote
// links create discrete-time recurrence. Exec.Backend on each Op is honored
// (BackendSIMD → DotTile/Saxpy).
//
// Tests live in github.com/openfluke/w2a — not here.
package step

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/runtime/backward"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/systems/tween"
)

// State holds the temporal snapshot of the 3D grid (loom StepState).
type State[T core.Numeric] struct {
	LayerData  []*core.Tensor[T]
	HistoryIn  [][]*core.Tensor[T]
	HistoryPre [][]*core.Tensor[T]
	NextBuffer []*core.Tensor[T]

	StepCount uint64
	mu        sync.RWMutex

	TweenState *tween.State[T]
	lastInput  *core.Tensor[T]
}

// New creates step state sized to g.StackLayerCount().
func New[T core.Numeric](g *architecture.Grid) *State[T] {
	n := 0
	if g != nil {
		n = g.StackLayerCount()
	}
	return &State[T]{
		LayerData:  make([]*core.Tensor[T], n),
		NextBuffer: make([]*core.Tensor[T], n),
		HistoryIn:  make([][]*core.Tensor[T], 0),
		HistoryPre: make([][]*core.Tensor[T], 0),
	}
}

// NewStepState is the loom-compatible alias.
func NewStepState[T core.Numeric](g *architecture.Grid) *State[T] { return New[T](g) }

// SetInput injects data into the starting linear index 0 (coord 0,0,0,0).
func (s *State[T]) SetInput(input *core.Tensor[T]) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastInput = input
	if len(s.LayerData) > 0 {
		s.LayerData[0] = input
	}
}

// Forward executes one clock cycle across the entire grid (parallel spatial tiles).
// Each enabled cell reads from LayerData (previous tick), writes NextBuffer.
func Forward[T core.Numeric](g *architecture.Grid, s *State[T], captureHistory bool) (time.Duration, error) {
	if g == nil || s == nil {
		return 0, fmt.Errorf("step: nil grid/state")
	}
	start := time.Now()
	s.mu.Lock()
	defer s.mu.Unlock()

	num := len(g.Cells)
	if len(s.LayerData) != num || len(s.NextBuffer) != num {
		s.LayerData = make([]*core.Tensor[T], num)
		s.NextBuffer = make([]*core.Tensor[T], num)
	}

	var currentIn, currentPre []*core.Tensor[T]
	if captureHistory {
		currentIn = make([]*core.Tensor[T], num)
		currentPre = make([]*core.Tensor[T], num)
	}

	var firstErr atomic.Value // error
	tile := 4
	if !g.Exec.MultiCore {
		tile = max(g.Depth, max(g.Rows, g.Cols)) // single tile → sequential
	}

	var wg sync.WaitGroup
	for zTile := 0; zTile < g.Depth; zTile += tile {
		zEnd := min(zTile+tile, g.Depth)
		for yTile := 0; yTile < g.Rows; yTile += tile {
			yEnd := min(yTile+tile, g.Rows)
			for xTile := 0; xTile < g.Cols; xTile += tile {
				xEnd := min(xTile+tile, g.Cols)
				wg.Add(1)
				go func(zT, zE, yT, yE, xT, xE int) {
					defer wg.Done()
					for z := zT; z < zE; z++ {
						for y := yT; y < yE; y++ {
							for x := xT; x < xE; x++ {
								for lIdx := 0; lIdx < g.LayersPerCell; lIdx++ {
									idx := g.Index(z, y, x, lIdx)
									cell := &g.Cells[idx]
									if cell.Layer.IsDisabled {
										if idx > 0 {
											s.NextBuffer[idx] = s.LayerData[idx-1]
										} else {
											s.NextBuffer[idx] = s.LayerData[idx]
										}
										continue
									}
									if cell.Op == nil {
										firstErr.Store(fmt.Errorf("step: no op at (%d,%d,%d,%d)", z, y, x, lIdx))
										continue
									}
									var input *core.Tensor[T]
									if cell.IsRemoteLink {
										tIdx := g.Index(cell.TargetZ, cell.TargetY, cell.TargetX, cell.TargetL)
										if tIdx >= 0 && tIdx < num {
											input = s.LayerData[tIdx]
										}
									} else if idx > 0 {
										input = s.LayerData[idx-1]
									} else {
										input = s.LayerData[0]
									}
									if input == nil {
										continue
									}
									pre, post, err := forward.Cell(cell, input)
									if err != nil {
										firstErr.Store(fmt.Errorf("step fwd (%d,%d,%d,%d): %w", z, y, x, lIdx, err))
										continue
									}
									s.NextBuffer[idx] = post
									if captureHistory {
										currentIn[idx] = input
										currentPre[idx] = pre
									}
								}
							}
						}
					}
				}(zTile, zEnd, yTile, yEnd, xTile, xEnd)
			}
		}
	}
	wg.Wait()

	if v := firstErr.Load(); v != nil {
		return time.Since(start), v.(error)
	}

	copy(s.LayerData, s.NextBuffer)
	if captureHistory {
		s.HistoryIn = append(s.HistoryIn, currentIn)
		s.HistoryPre = append(s.HistoryPre, currentPre)
	}
	s.StepCount++
	return time.Since(start), nil
}

// StepForward is the loom-compatible alias.
func StepForward[T core.Numeric](g *architecture.Grid, s *State[T], captureHistory bool) (time.Duration, error) {
	return Forward(g, s, captureHistory)
}

// Backward propagates gradients through captured step history (BPTT across ticks).
func Backward[T core.Numeric](g *architecture.Grid, s *State[T], gradOutput *core.Tensor[T]) (
	gradIn *core.Tensor[T], layerGradients [][2]*core.Tensor[T], err error,
) {
	if g == nil || s == nil {
		return nil, nil, fmt.Errorf("step: nil grid/state")
	}
	s.mu.RLock()
	defer s.mu.RUnlock()

	if len(s.HistoryIn) == 0 {
		return nil, nil, nil
	}

	numSteps := len(s.HistoryIn)
	numLayers := len(g.Cells)
	layerGradients = make([][2]*core.Tensor[T], numLayers)
	gradBuffers := make([]*core.Tensor[T], numLayers)
	gradBuffers[numLayers-1] = gradOutput

	for stepIdx := numSteps - 1; stepIdx >= 0; stepIdx-- {
		stepIn := s.HistoryIn[stepIdx]
		stepPre := s.HistoryPre[stepIdx]
		nextGradBuffers := make([]*core.Tensor[T], numLayers)

		for idx := numLayers - 1; idx >= 0; idx-- {
			cell := &g.Cells[idx]
			if cell.Layer.IsDisabled {
				if gradBuffers[idx] != nil {
					accumulateMeshGrad(g, nextGradBuffers, idx, gradBuffers[idx])
				}
				continue
			}
			input := stepIn[idx]
			pre := stepPre[idx]
			currentGrad := gradBuffers[idx]
			if input == nil || pre == nil || currentGrad == nil {
				continue
			}
			gIn, gW, err := backward.Cell(cell, currentGrad, input, pre)
			if err != nil {
				return nil, nil, fmt.Errorf("step bwd idx=%d: %w", idx, err)
			}
			if layerGradients[idx][1] == nil {
				layerGradients[idx] = [2]*core.Tensor[T]{gIn, gW}
			} else if gW != nil && layerGradients[idx][1] != nil {
				for i := range layerGradients[idx][1].Data {
					if i < len(gW.Data) {
						layerGradients[idx][1].Data[i] += gW.Data[i]
					}
				}
			}
			if gIn != nil {
				accumulateMeshGrad(g, nextGradBuffers, idx, gIn)
			}
		}
		gradBuffers = nextGradBuffers
	}
	return gradBuffers[0], layerGradients, nil
}

// StepBackward is the loom-compatible alias.
func StepBackward[T core.Numeric](g *architecture.Grid, s *State[T], gradOutput *core.Tensor[T]) (
	gradIn *core.Tensor[T], layerGradients [][2]*core.Tensor[T], err error,
) {
	return Backward(g, s, gradOutput)
}

// ApplyTween bridges mesh activations into tween gap updates (layerwise by default).
func ApplyTween[T core.Numeric](g *architecture.Grid, s *State[T], globalTarget *core.Tensor[T], lr float32) error {
	if g == nil || s == nil || globalTarget == nil {
		return fmt.Errorf("step: ApplyTween nil args")
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.TweenState == nil {
		cfg := tween.DefaultConfig()
		cfg.UseChainRule = false // gap-based for step mesh (loom default)
		s.TweenState = tween.NewState[T](g, cfg)
	}

	if len(s.LayerData) > 0 {
		s.TweenState.ForwardActs[0] = s.lastInput
		for i := 0; i < len(s.LayerData) && i+1 < len(s.TweenState.ForwardActs); i++ {
			s.TweenState.ForwardActs[i+1] = s.LayerData[i]
		}
	}

	if err := tween.Backward(g, s.TweenState, globalTarget); err != nil {
		return err
	}
	s.TweenState.CalculateLinkBudgets()
	return tween.ApplyGaps(g, s.TweenState, lr)
}

// StepApplyTween is the loom-compatible alias.
func StepApplyTween[T core.Numeric](g *architecture.Grid, s *State[T], globalTarget *core.Tensor[T], lr float32) error {
	return ApplyTween(g, s, globalTarget, lr)
}

func accumulateMeshGrad[T core.Numeric](g *architecture.Grid, buffers []*core.Tensor[T], idx int, grad *core.Tensor[T]) {
	if g == nil || grad == nil || idx < 0 || idx >= len(g.Cells) {
		return
	}
	cell := &g.Cells[idx]
	var sourceIdx int
	if cell.IsRemoteLink {
		sourceIdx = g.Index(cell.TargetZ, cell.TargetY, cell.TargetX, cell.TargetL)
	} else if idx > 0 {
		sourceIdx = idx - 1
	} else {
		sourceIdx = 0
	}
	if sourceIdx < 0 || sourceIdx >= len(buffers) {
		return
	}
	if buffers[sourceIdx] == nil {
		buffers[sourceIdx] = grad.Clone()
		return
	}
	dst := buffers[sourceIdx]
	n := dst.Len()
	if grad.Len() < n {
		n = grad.Len()
	}
	for i := 0; i < n; i++ {
		dst.Data[i] += grad.Data[i]
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
