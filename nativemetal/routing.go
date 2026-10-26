package nativemetal

import (
	"bytes"
	"fmt"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
	"runtime"
	"sync"
)

// RoutingStats counts successful native dispatches; useful for checking placement.
type RoutingStats struct{ Mounts, Forwards, Backwards, Updates, TrainingSteps uint64 }
type routedModel struct {
	session    *Session
	descriptor []byte
}

// Router owns persistent per-operation Metal mounts for the normal Go executor.
// The executor transfers Float32 activations/gradients across backend boundaries.
// Parameters are synchronized automatically after native updates, so existing Go
// saving and switching backends observe the current weights. Explicit Mount
// sessions retain their separate, explicit SyncToGo contract.
type Router struct {
	mu         sync.Mutex
	options    []Options
	models     map[any]*routedModel
	unregister func()
	closed     bool
	stats      RoutingStats
}

// EnableRouting installs automatic Exec.Backend/JSON Metal routing. Keep the
// returned owner alive and close it when the model workload ends. Only one
// automatic router is active at a time; explicit sessions remain independent.
func EnableRouting(options ...Options) (*Router, error) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		return nil, fmt.Errorf("nativemetal: routing requires Apple silicon macOS")
	}
	if len(options) > 1 {
		return nil, fmt.Errorf("nativemetal: at most one Options value")
	}
	r := &Router{options: options, models: make(map[any]*routedModel)}
	unregister, err := core.RegisterMetalExecutor(r)
	if err != nil {
		return nil, err
	}
	r.unregister = unregister
	return r, nil
}
func (r *Router) model(op any) (*routedModel, error) {
	if r.closed {
		return nil, fmt.Errorf("nativemetal: router closed")
	}
	// A previous C++ training call may still own pending parameter updates.
	if pending, ok := op.(interface{ NativeWeightsPending() bool }); ok && pending.NativeWeightsPending() {
		syncer, ok := op.(interface{ SyncWeightsFromNative() error })
		if !ok {
			return nil, fmt.Errorf("nativemetal: pending native weights cannot be synchronized")
		}
		if err := syncer.SyncWeightsFromNative(); err != nil {
			return nil, err
		}
	}
	descriptor, _, err := encodeModel(op)
	if err != nil {
		return nil, err
	}
	m := r.models[op]
	if m != nil && bytes.Equal(m.descriptor, descriptor) {
		return m, nil
	}
	s, err := Mount(op, r.options...)
	if err != nil {
		return nil, err
	}
	if m != nil {
		_ = m.session.Close()
	}
	m = &routedModel{session: s, descriptor: descriptor}
	r.models[op] = m
	r.stats.Mounts++
	return m, nil
}
func (r *Router) sync(op any, m *routedModel) error {
	if err := m.session.SyncToGo(); err != nil {
		return err
	}
	p, _, err := encodeModel(op)
	if err != nil {
		return err
	}
	m.descriptor = p
	return nil
}
func (r *Router) Forward(op any, x *core.Tensor[float32]) (*core.Tensor[float32], *core.Tensor[float32], error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	m, err := r.model(op)
	if err != nil {
		return nil, nil, err
	}
	pre, post, err := m.session.ForwardPair(x)
	if err != nil {
		return nil, nil, err
	}
	// Meta and other stateful forwards may update parameters as part of execution.
	if err = r.sync(op, m); err != nil {
		return nil, nil, err
	}
	r.stats.Forwards++
	return pre, post, nil
}
func (r *Router) Backward(op any, gy, x *core.Tensor[float32]) (*core.Tensor[float32], *core.Tensor[float32], error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	m, err := r.model(op)
	if err != nil {
		return nil, nil, err
	}
	gx, dw, err := m.session.Backward(x, gy)
	if err != nil {
		return nil, nil, err
	}
	if err = r.sync(op, m); err != nil {
		return nil, nil, err
	}
	r.stats.Backwards++
	return gx, dw, nil
}
func (r *Router) ApplySGD(op any, dw *core.Tensor[float32], lr float64) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	m, err := r.model(op)
	if err != nil {
		return err
	}
	if err = m.session.ApplySGD(dw, lr); err != nil {
		return err
	}
	if err = r.sync(op, m); err != nil {
		return err
	}
	r.stats.Updates++
	return nil
}
func (r *Router) TrainMSE(op any, x, target *core.Tensor[float32], mode int, lr float64) (float64, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	m, err := r.model(op)
	if err != nil {
		return 0, err
	}
	loss, err := m.session.TrainMSE(x, target, parallel.TrainMode(mode), lr)
	if err != nil {
		return 0, err
	}
	if err = r.sync(op, m); err != nil {
		return 0, err
	}
	r.stats.TrainingSteps++
	return loss, nil
}
func (r *Router) Stats() RoutingStats { r.mu.Lock(); defer r.mu.Unlock(); return r.stats }

// Close unregisters dispatch before releasing sessions and is idempotent.
func (r *Router) Close() error {
	if r == nil {
		return nil
	}
	r.unregister()
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil
	}
	r.closed = true
	for op, m := range r.models {
		_ = m.session.Close()
		delete(r.models, op)
	}
	return nil
}

func (r *Router) DenseProxy(op any, gy, x, pre *core.Tensor[float32], kind int) (*core.Tensor[float32], error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	m, err := r.model(op)
	if err != nil {
		return nil, err
	}
	out, err := m.session.denseProxy(x, gy, pre, kind)
	if err == nil {
		r.stats.Backwards++
	}
	return out, err
}

func (r *Router) TrainCredit(op any, gy, x, pre *core.Tensor[float32], mode int, lr float64) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	m, err := r.model(op)
	if err != nil {
		return err
	}
	m.session.mu.Lock()
	var e encoder
	encodeTensor(&e, x)
	encodeTensor(&e, gy)
	if pre == nil {
		e.ints(0)
	} else {
		e.ints(1)
		encodeTensor(&e, pre)
	}
	if e.err == nil {
		_, err = m.session.d.segment(mode+10, lr, e.Bytes())
	} else {
		err = e.err
	}
	m.session.mu.Unlock()
	if err != nil {
		return err
	}
	if err = r.sync(op, m); err != nil {
		return err
	}
	r.stats.TrainingSteps++
	return nil
}
