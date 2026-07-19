// Package metacognition wraps an observed Dense with heuristic stability rules (loom Meta).
//
// Welvet policy: no QAT / dtype morph-as-training. Rules may gate, scale, or reset
// observed weights toward identity — they do not morph LayerType.
// Contract: CPU tiled + SIMD + WebGPU via Observed Exec. Tests in w2a.
package metacognition
