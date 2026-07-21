// Package wav2vec2 runs facebook/wav2vec2-base-960h (Wav2Vec2ForCTC) in pure Go.
//
// Scope (v0): FP32 greedy CTC transcription from 16 kHz mono PCM / WAV.
// Feature extractor, convolutional positional embedding, encoder, and lm_head
// match Hugging Face transformers inference (dropout/layerdrop off).
package wav2vec2
