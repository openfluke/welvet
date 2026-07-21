package wav2vec2

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

// ReadWAVMono loads 16-bit PCM WAV → float32 mono in [-1, 1] + sample rate.
// Borrowed pattern from apps/mosstts (no package dependency).
func ReadWAVMono(path string) ([]float32, int, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, err
	}
	if len(data) < 44 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("wav2vec2: not WAV")
	}
	var sampleRate, channels, bits int
	var pcm []byte
	i := 12
	for i+8 <= len(data) {
		chunk := string(data[i : i+4])
		sz := int(binary.LittleEndian.Uint32(data[i+4 : i+8]))
		body := i + 8
		if body+sz > len(data) {
			break
		}
		if chunk == "fmt " && sz >= 16 {
			channels = int(binary.LittleEndian.Uint16(data[body+2 : body+4]))
			sampleRate = int(binary.LittleEndian.Uint32(data[body+4 : body+8]))
			bits = int(binary.LittleEndian.Uint16(data[body+14 : body+16]))
		} else if chunk == "data" {
			pcm = data[body : body+sz]
			break
		}
		i = body + sz
		if sz%2 == 1 {
			i++
		}
	}
	if pcm == nil || sampleRate == 0 || channels == 0 || bits != 16 {
		return nil, 0, fmt.Errorf("wav2vec2: unsupported WAV sr=%d ch=%d bits=%d", sampleRate, channels, bits)
	}
	n := len(pcm) / 2
	frames := n / channels
	out := make([]float32, frames)
	for f := 0; f < frames; f++ {
		var sum float32
		for c := 0; c < channels; c++ {
			s := int16(binary.LittleEndian.Uint16(pcm[(f*channels+c)*2:]))
			sum += float32(s) / 32768
		}
		out[f] = sum / float32(channels)
	}
	return out, sampleRate, nil
}

// ResampleLinear resamples mono audio to targetHz with linear interpolation.
func ResampleLinear(x []float32, srcHz, dstHz int) []float32 {
	if srcHz == dstHz || len(x) == 0 {
		return append([]float32(nil), x...)
	}
	nOut := int(float64(len(x)) * float64(dstHz) / float64(srcHz))
	if nOut < 1 {
		nOut = 1
	}
	out := make([]float32, nOut)
	scale := float64(srcHz) / float64(dstHz)
	for i := range out {
		src := float64(i) * scale
		j := int(src)
		frac := float32(src - float64(j))
		if j >= len(x)-1 {
			out[i] = x[len(x)-1]
			continue
		}
		out[i] = x[j]*(1-frac) + x[j+1]*frac
	}
	return out
}

// NormalizeWaveform zero-means and unit-variances audio (HF Wav2Vec2Processor do_normalize).
func NormalizeWaveform(x []float32) []float32 {
	if len(x) == 0 {
		return x
	}
	var sum float64
	for _, v := range x {
		sum += float64(v)
	}
	mean := sum / float64(len(x))
	var varSum float64
	for _, v := range x {
		d := float64(v) - mean
		varSum += d * d
	}
	std := math.Sqrt(varSum/float64(len(x)) + 1e-7)
	out := make([]float32, len(x))
	inv := 1 / std
	for i, v := range x {
		out[i] = float32((float64(v) - mean) * inv)
	}
	return out
}

// PrepareAudio loads WAV, resamples to 16 kHz, and applies HF normalize.
func PrepareAudio(path string) ([]float32, error) {
	x, sr, err := ReadWAVMono(path)
	if err != nil {
		return nil, err
	}
	if sr != 16000 {
		x = ResampleLinear(x, sr, 16000)
	}
	return NormalizeWaveform(x), nil
}
