// Package parallel is Parallel / MoE combine (loom Parallel) plus Stack for
// nested multi-cameral graphs with TrainMode (SGD / Tween / TweenChain / TweenSplit /
// Freeze / Shadow / Adversarial / Memory) and optional per-hemisphere BranchModes /
// per-child ChildModes.
//
// Parallel branches are polymorphic Ops. Combine modes: concat (default), add,
// avg, filter (MoE gate), max, sparsek, disagree. Stack is a heterogeneous Op chain —
// use Hemispheres / CameralFromBranches / Bicameral / BicameralFrom / Sandwich for
// cameral graphs over any Parallel-supported layer kind.
// Train / TrainStack / TrainStackMSE / TrainStackCE drive updates. Freeze/Shadow
// still forward and combine but skip weight updates (CamSync skips frozen cams).
// CamKit adds Shadow KD, DNAReg, surprise Memory, DreamPulse, plasticity metrics;
// BranchLRs + RotateSchedule control per-cam step size and sleep cycles.
// Step* uses a 1D line pipe (one child hop per tick). PlaceStack + runtime/step is the
// volumetric mesh clock. Mesh* still needs a Grid.
// Contract: CPU tiled + SIMD + WebGPU via children Exec; dtype × k-quant on
// branches/gate. No QAT. Tests live in github.com/openfluke/w2a — not here.
//
// CamSync (CamSyncConfig) optionally averages cameral branch weights after
// sample / step / pulse (alpha 1%→100%, BranchAlpha asymmetric, groups + cross pairs).
package parallel
