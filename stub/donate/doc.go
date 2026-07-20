// Package donate is the networked compute-offload protocol (loom donate_compute_*).
//
// TCP frames: u32 LE length + JSON. Modes: model_push | local_lm.
// v0 workers stub-echo infer/prompt. Tests in github.com/openfluke/w2a.
package donate
