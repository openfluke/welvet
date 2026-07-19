// Package tanhi streams sparse non-blocking JSON-line UDP events for HUD visualization
// (loom/poly TANHI rebuild). Best-effort queue — overflow drops so training never blocks.
//
// Tests live in github.com/openfluke/w2a — not here.
package tanhi

import (
	"encoding/json"
	"fmt"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/systems/dna"
)

// DefaultUDPPort is the default client destination (IANA unassigned range).
const DefaultUDPPort = 17481

// UDPConfig controls JSON line telemetry to a UDP listener (e.g. SoulGlitch TANHI HUD).
type UDPConfig struct {
	Enabled   bool
	Host      string
	Port      int
	SendShape bool
}

// Alias matching loom naming.
type TanhiUDPConfig = UDPConfig

// DefaultTanhiUDPPort mirrors loom constant name.
const DefaultTanhiUDPPort = DefaultUDPPort

type coordWire struct {
	Z int `json:"z"`
	Y int `json:"y"`
	X int `json:"x"`
	L int `json:"l"`
}

type branchWire struct {
	Type   string `json:"type"`
	Slot   int    `json:"slot"`
	Remote bool   `json:"remote,omitempty"`
	TZ     int    `json:"tz,omitempty"`
	TY     int    `json:"ty,omitempty"`
	TX     int    `json:"tx,omitempty"`
	TL     int    `json:"tl,omitempty"`
}

type wireEvent struct {
	V           string       `json:"v"`
	Seq         uint64       `json:"seq"`
	Phase       string       `json:"phase"`
	Idx         int          `json:"idx"`
	Z           int          `json:"z"`
	Y           int          `json:"y"`
	X           int          `json:"x"`
	L           int          `json:"l"`
	Layer       string       `json:"layer"`
	DType       int          `json:"dtype"`
	Connections int          `json:"connections"`
	T0Ns        int64        `json:"t0_ns"`
	T1Ns        int64        `json:"t1_ns"`
	Shape       []int        `json:"shape,omitempty"`
	Links       []coordWire  `json:"links,omitempty"`
	Label       string       `json:"label,omitempty"`
	Combine     string       `json:"combine,omitempty"`
	Branches    []branchWire `json:"branches,omitempty"`
}

var (
	seq      atomic.Uint64
	queue    = make(chan packet, 1024)
	initOnce sync.Once
	addrMu   sync.RWMutex
	addrMap  = make(map[string]*net.UDPAddr)
)

type packet struct {
	addr *net.UDPAddr
	data []byte
}

func writerLoop() {
	var pc net.PacketConn
	for pkt := range queue {
		if pc == nil {
			c, err := net.ListenPacket("udp", ":0")
			if err != nil {
				continue
			}
			pc = c
		}
		_, _ = pc.WriteTo(pkt.data, pkt.addr)
	}
}

func ensureWriter() {
	initOnce.Do(func() {
		go writerLoop()
	})
}

func resolveUDPAddr(cfg *UDPConfig) *net.UDPAddr {
	if cfg == nil {
		return nil
	}
	host := cfg.Host
	if host == "" {
		host = "127.0.0.1"
	}
	port := cfg.Port
	if port == 0 {
		port = DefaultUDPPort
	}
	key := fmt.Sprintf("%s:%d", host, port)
	addrMu.RLock()
	a, ok := addrMap[key]
	addrMu.RUnlock()
	if ok {
		return a
	}
	addr, err := net.ResolveUDPAddr("udp", key)
	if err != nil {
		return nil
	}
	addrMu.Lock()
	addrMap[key] = addr
	addrMu.Unlock()
	return addr
}

func connectionCount(cell *architecture.Cell) int {
	if cell == nil {
		return 0
	}
	flat, err := dna.FlattenOp(cell.Op)
	if err != nil {
		return 0
	}
	return len(flat)
}

func routingLinks(cell *architecture.Cell) []coordWire {
	if cell == nil {
		return nil
	}
	seen := make(map[coordWire]struct{})
	var out []coordWire
	add := func(c coordWire) {
		if _, ok := seen[c]; ok {
			return
		}
		seen[c] = struct{}{}
		out = append(out, c)
	}
	for i := range cell.ParallelBranches {
		b := &cell.ParallelBranches[i]
		if b.IsRemoteLink {
			add(coordWire{Z: b.TargetZ, Y: b.TargetY, X: b.TargetX, L: b.TargetL})
		} else {
			add(coordWire{Z: b.Layer.Z, Y: b.Layer.Y, X: b.Layer.X, L: b.Layer.L})
		}
	}
	for i := range cell.SequentialLayers {
		b := &cell.SequentialLayers[i]
		if b.IsRemoteLink {
			add(coordWire{Z: b.TargetZ, Y: b.TargetY, X: b.TargetX, L: b.TargetL})
		} else {
			add(coordWire{Z: b.Layer.Z, Y: b.Layer.Y, X: b.Layer.X, L: b.Layer.L})
		}
	}
	const maxLinks = 48
	if len(out) > maxLinks {
		out = out[:maxLinks]
	}
	return out
}

func branchList(cell *architecture.Cell) []branchWire {
	if cell == nil {
		return nil
	}
	var out []branchWire
	appendBranch := func(slot int, b *architecture.Cell) {
		if b == nil {
			return
		}
		bw := branchWire{Type: b.Layer.Type.String(), Slot: slot}
		if b.IsRemoteLink {
			bw.Remote = true
			bw.TZ, bw.TY, bw.TX, bw.TL = b.TargetZ, b.TargetY, b.TargetX, b.TargetL
		}
		out = append(out, bw)
	}
	for i := range cell.ParallelBranches {
		appendBranch(i, &cell.ParallelBranches[i])
	}
	for i := range cell.SequentialLayers {
		appendBranch(i, &cell.SequentialLayers[i])
	}
	const maxBranches = 32
	if len(out) > maxBranches {
		out = out[:maxBranches]
	}
	return out
}

// GPULayerShapeHint returns an approximate tensor shape for telemetry (no GPU readback).
func GPULayerShapeHint(meta core.Layer, numTokens int) []int {
	if numTokens <= 0 {
		return nil
	}
	switch meta.Type {
	case core.LayerRMSNorm, core.LayerLayerNorm:
		return []int{numTokens, meta.InputHeight}
	case core.LayerMultiHeadAttention:
		d := meta.OutputHeight
		if d <= 0 {
			d = meta.InputHeight
		}
		return []int{numTokens, d}
	case core.LayerDense, core.LayerSwiGLU, core.LayerEmbedding:
		if meta.OutputHeight > 0 {
			return []int{numTokens, meta.OutputHeight}
		}
		return []int{numTokens, meta.InputHeight}
	default:
		if meta.OutputHeight > 0 {
			return []int{numTokens, meta.OutputHeight}
		}
		return []int{numTokens, meta.InputHeight}
	}
}

// TanhiGPULayerShapeHint is the loom-compatible alias.
func TanhiGPULayerShapeHint(meta core.Layer, numTokens int) []int {
	return GPULayerShapeHint(meta, numTokens)
}

// EmitSweep sends a run-boundary marker so HUDs reset between configs.
func EmitSweep(cfg *UDPConfig, label string) {
	if cfg == nil || !cfg.Enabled || label == "" {
		return
	}
	addr := resolveUDPAddr(cfg)
	if addr == nil {
		return
	}
	now := time.Now().UnixNano()
	w := wireEvent{
		V:     "tanhi1",
		Seq:   seq.Add(1),
		Phase: "sweep",
		Layer: "sweep",
		Label: label,
		T0Ns:  now,
		T1Ns:  now,
	}
	enqueue(addr, w)
}

// TanhiEmitSweep is the loom-compatible alias.
func TanhiEmitSweep(cfg *UDPConfig, label string) { EmitSweep(cfg, label) }

// ConfigFromGrid extracts *UDPConfig from grid.Tanhi (any).
func ConfigFromGrid(g *architecture.Grid) *UDPConfig {
	if g == nil || g.Tanhi == nil {
		return nil
	}
	switch v := g.Tanhi.(type) {
	case *UDPConfig:
		return v
	case UDPConfig:
		cp := v
		return &cp
	default:
		return nil
	}
}

// Emit records one layer boundary event (non-blocking; drops if queue full).
func Emit(cfg *UDPConfig, phase string, idx int, cell *architecture.Cell, t0, t1 time.Time, shape []int) {
	EmitWithConn(cfg, phase, idx, cell, t0, t1, shape, -1)
}

// EmitWithConn allows overriding the connection count (e.g. tied LM head).
func EmitWithConn(cfg *UDPConfig, phase string, idx int, cell *architecture.Cell, t0, t1 time.Time, shape []int, connOverride int) {
	if cfg == nil || !cfg.Enabled || cell == nil {
		return
	}
	addr := resolveUDPAddr(cfg)
	if addr == nil {
		return
	}
	conn := connectionCount(cell)
	if connOverride >= 0 {
		conn = connOverride
	}
	w := wireEvent{
		V:           "tanhi1",
		Seq:         seq.Add(1),
		Phase:       phase,
		Idx:         idx,
		Z:           cell.Layer.Z,
		Y:           cell.Layer.Y,
		X:           cell.Layer.X,
		L:           cell.Layer.L,
		Layer:       cell.Layer.Type.String(),
		DType:       int(cell.Layer.DType),
		Connections: conn,
		T0Ns:        t0.UnixNano(),
		T1Ns:        t1.UnixNano(),
	}
	if cfg.SendShape && len(shape) > 0 {
		w.Shape = shape
	}
	if links := routingLinks(cell); len(links) > 0 {
		w.Links = links
	}
	if branches := branchList(cell); len(branches) > 0 {
		w.Branches = branches
	}
	if cell.Layer.Type == core.LayerParallel && cell.CombineMode != "" {
		w.Combine = cell.CombineMode
	}
	enqueue(addr, w)
}

func enqueue(addr *net.UDPAddr, w wireEvent) {
	line, err := json.Marshal(w)
	if err != nil {
		return
	}
	line = append(line, '\n')
	ensureWriter()
	select {
	case queue <- packet{addr: addr, data: line}:
	default:
	}
}
