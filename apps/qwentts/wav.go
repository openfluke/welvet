package qwentts

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

// WriteWAV writes float32 PCM as 16-bit little-endian PCM WAV.
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
