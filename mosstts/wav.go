package mosstts

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

// WriteWAV writes little-endian PCM float32 interleaved stereo (or mono) as 16-bit PCM WAV.
func WriteWAV(path string, samples []float32, sampleRate, channels int) error {
	if channels < 1 {
		channels = 1
	}
	nFrames := len(samples) / channels
	if nFrames*channels != len(samples) {
		return fmt.Errorf("WriteWAV: samples len %d not divisible by channels %d", len(samples), channels)
	}
	pcm := make([]byte, nFrames*channels*2)
	for i, v := range samples {
		if v > 1 {
			v = 1
		} else if v < -1 {
			v = -1
		}
		s := int16(math.Round(float64(v) * 32767))
		binary.LittleEndian.PutUint16(pcm[i*2:], uint16(s))
	}
	dataLen := len(pcm)
	buf := make([]byte, 44+dataLen)
	copy(buf[0:], []byte("RIFF"))
	binary.LittleEndian.PutUint32(buf[4:], uint32(36+dataLen))
	copy(buf[8:], []byte("WAVE"))
	copy(buf[12:], []byte("fmt "))
	binary.LittleEndian.PutUint32(buf[16:], 16)
	binary.LittleEndian.PutUint16(buf[20:], 1) // PCM
	binary.LittleEndian.PutUint16(buf[22:], uint16(channels))
	binary.LittleEndian.PutUint32(buf[24:], uint32(sampleRate))
	byteRate := sampleRate * channels * 2
	binary.LittleEndian.PutUint32(buf[28:], uint32(byteRate))
	binary.LittleEndian.PutUint16(buf[32:], uint16(channels*2))
	binary.LittleEndian.PutUint16(buf[34:], 16)
	copy(buf[36:], []byte("data"))
	binary.LittleEndian.PutUint32(buf[40:], uint32(dataLen))
	copy(buf[44:], pcm)
	return os.WriteFile(path, buf, 0o644)
}

// ReadWAVMono loads 16-bit PCM WAV and returns float32 mono (mean of channels) + sample rate.
func ReadWAVMono(path string) ([]float32, int, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, err
	}
	if len(data) < 44 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("ReadWAV: not WAV")
	}
	// naive parse: find "fmt " and "data"
	var sampleRate, channels, bits int
	var pcm []byte
	i := 12
	for i+8 <= len(data) {
		chunk := string(data[i : i+4])
		sz := int(binary.LittleEndian.Uint32(data[i+4 : i+8]))
		body := i + 8
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
		return nil, 0, fmt.Errorf("ReadWAV: unsupported format sr=%d ch=%d bits=%d", sampleRate, channels, bits)
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
