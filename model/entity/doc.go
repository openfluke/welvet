// Package entity is Welvet's native .entity checkpoint format (ENTITY magic + JSON + blobs).
//
// Primary APIs:
//   - Open / Inspect / LoadBlob / LoadQuantBlob — read packed transformer checkpoints
//   - PackFromHF / ImportFromHF — Hugging Face snapshot → .entity
//   - WriteTransformerFile — write a packed transformer checkpoint
//   - SerializeNetwork / ParseNetwork — volumetric-grid ENTITY (used by stub/serialization)
//
// Tests live in w2a — not here.
package entity

const (
	// Magic is the 8-byte file signature (Loom wire layout).
	Magic         = "ENTITY\x00\x00"
	FormatVersion = 1
	headerMaxSize = 256 << 20
	// TokenizerBlobPath is the UTF-8 tokenizer.json payload (portable; no host paths).
	TokenizerBlobPath = "transformer.tokenizer.json"
)

func fixedHeaderSize() int { return 20 }
