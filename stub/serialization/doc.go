// Package serialization is JSON + ENTITY checkpoint I/O for volumetric grids.
//
// Storage truth: FormatNone × dtype uses native bytes (LE f32 for float32);
// block quants persist EncodePackedWire(Packed) — never Flatten→re-Pack as truth.
// Welvet does not do QAT.
//
// Tests live in github.com/openfluke/w2a — not here.
package serialization
