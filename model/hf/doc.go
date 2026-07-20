// Package hf imports Hugging Face snapshots (config.json + safetensors)
// into Welvet-native structures for entity packing.
//
// Primary APIs:
//   - InspectSnapshot — probe arch / dims / EOS without loading weights
//   - DetectArchitecture / ParseDecoderDims / LoadSafetensorsSelective — building blocks
//   - entity.PackFromHF / entity.ImportFromHF — bake Welvet .entity checkpoints
//
// Tests live in github.com/openfluke/w2a — not here.
package hf
