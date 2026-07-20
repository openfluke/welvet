package observer

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"sync"
	"time"

	"github.com/openfluke/welvet/core"
)

// Observer defines the interface for tracking neural activity.
type Observer interface {
	OnForward(event LayerEvent)
	OnBackward(event LayerEvent)
}

// LayerEvent captures state during a forward or backward pass.
type LayerEvent struct {
	Mode      string         `json:"mode"`
	Type      string         `json:"type"`
	Z         int            `json:"z"`
	Y         int            `json:"y"`
	X         int            `json:"x"`
	L         int            `json:"l"`
	LayerType core.LayerType `json:"layer_type"`
	Stats     LayerStats     `json:"stats"`
	StepCount uint64         `json:"step_count"`
	ModelID   string         `json:"model_id"`
}

// LayerStats provides summary statistics for a tensor.
type LayerStats struct {
	Avg    float32 `json:"avg"`
	Std    float32 `json:"std"`
	Max    float32 `json:"max"`
	Min    float32 `json:"min"`
	Active int     `json:"active"`
	Total  int     `json:"total"`
}

// ComputeLayerStats calculates summary statistics for a tensor.
func ComputeLayerStats[T core.Numeric](t *core.Tensor[T]) LayerStats {
	if t == nil || len(t.Data) == 0 {
		return LayerStats{}
	}
	var sum, sumSq, max, min float32
	max = float32(core.AsFloat64(t.Data[0]))
	min = float32(core.AsFloat64(t.Data[0]))
	active := 0
	for _, v := range t.Data {
		f := float32(core.AsFloat64(v))
		sum += f
		sumSq += f * f
		if f > max {
			max = f
		}
		if f < min {
			min = f
		}
		if math.Abs(float64(f)) > 1e-6 {
			active++
		}
	}
	n := float32(len(t.Data))
	avg := sum / n
	variance := (sumSq / n) - (avg * avg)
	if variance < 0 {
		variance = 0
	}
	return LayerStats{
		Avg: avg, Std: float32(math.Sqrt(float64(variance))),
		Max: max, Min: min, Active: active, Total: len(t.Data),
	}
}

// ConsoleObserver prints events to stdout.
type ConsoleObserver struct{}

func (o *ConsoleObserver) OnForward(e LayerEvent) {
	fmt.Printf("[FWD] (%d,%d,%d,%d) %v: avg=%.4f max=%.4f\n",
		e.Z, e.Y, e.X, e.L, e.LayerType, e.Stats.Avg, e.Stats.Max)
}

func (o *ConsoleObserver) OnBackward(e LayerEvent) {
	fmt.Printf("[BWD] (%d,%d,%d,%d) %v: grad_avg=%.4f\n",
		e.Z, e.Y, e.X, e.L, e.LayerType, e.Stats.Avg)
}

// HTTPObserver sends events to an HTTP endpoint.
type HTTPObserver struct {
	URL    string
	client *http.Client
}

// NewHTTPObserver creates an observer with a short POST timeout.
func NewHTTPObserver(url string) *HTTPObserver {
	return &HTTPObserver{URL: url, client: &http.Client{Timeout: 100 * time.Millisecond}}
}

func (o *HTTPObserver) OnForward(e LayerEvent)  { o.send(e) }
func (o *HTTPObserver) OnBackward(e LayerEvent) { o.send(e) }

func (o *HTTPObserver) send(e LayerEvent) {
	data, _ := json.Marshal(e)
	go func() {
		resp, err := o.client.Post(o.URL, "application/json", bytes.NewReader(data))
		if err == nil && resp != nil {
			resp.Body.Close()
		}
	}()
}

// BufferObserver collects events in memory.
type BufferObserver struct {
	WindowSize int
	History    []LayerStats
	Events     []LayerEvent
	mu         sync.Mutex
}

// NewBufferObserver creates a buffer with optional rolling window aggregation.
func NewBufferObserver(windowSize int) *BufferObserver {
	return &BufferObserver{WindowSize: windowSize, Events: make([]LayerEvent, 0)}
}

func (o *BufferObserver) OnForward(e LayerEvent) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.Events = append(o.Events, e)
	if o.WindowSize > 0 && len(o.Events) >= o.WindowSize {
		var avg, std, max, min float32
		for _, ev := range o.Events {
			avg += ev.Stats.Avg
			std += ev.Stats.Std
			if ev.Stats.Max > max {
				max = ev.Stats.Max
			}
			if ev.Stats.Min < min {
				min = ev.Stats.Min
			}
		}
		n := float32(len(o.Events))
		o.History = append(o.History, LayerStats{Avg: avg / n, Std: std / n, Max: max, Min: min, Total: e.Stats.Total})
		o.Events = nil
	}
}

func (o *BufferObserver) OnBackward(e LayerEvent) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.Events = append(o.Events, e)
}

// Snapshot returns a copy of buffered events.
func (o *BufferObserver) Snapshot() []LayerEvent {
	o.mu.Lock()
	defer o.mu.Unlock()
	out := make([]LayerEvent, len(o.Events))
	copy(out, o.Events)
	return out
}
