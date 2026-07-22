package qwenasr

import "math"

const (
	melFFT  = 400
	melHop  = 160
	melBins = 128
)

func hzMel(h float64) float64 {
	if h < 1000 {
		return 3 * h / 200
	}
	return 15 + math.Log(h/1000)*27/math.Log(6.4)
}
func melHz(m float64) float64 {
	if m < 15 {
		return 200 * m / 3
	}
	return 1000 * math.Exp((m-15)*math.Log(6.4)/27)
}
func melFilter() [][]float64 {
	f := make([][]float64, melBins)
	pts := make([]float64, melBins+2)
	lo, hi := hzMel(0), hzMel(8000)
	for i := range pts {
		pts[i] = melHz(lo + (hi-lo)*float64(i)/float64(melBins+1))
	}
	for m := range f {
		f[m] = make([]float64, 201)
		norm := 2 / (pts[m+2] - pts[m])
		for k := 0; k <= 200; k++ {
			hz := 8000 * float64(k) / 200
			down := (hz - pts[m]) / (pts[m+1] - pts[m])
			up := (pts[m+2] - hz) / (pts[m+2] - pts[m+1])
			if down < up {
				f[m][k] = math.Max(0, down) * norm
			} else {
				f[m][k] = math.Max(0, up) * norm
			}
		}
	}
	return f
}

// melSpectrogram matches WhisperFeatureExtractor: centered Hann STFT, final
// frame dropped, then Slaney mel and dynamic-range normalization.
func melSpectrogram(pcm []float32) []float32 {
	pad := melFFT / 2
	x := make([]float32, len(pcm)+2*pad)
	for i := range x {
		j := i - pad
		if j < 0 {
			j = -j
		}
		if j >= len(pcm) {
			j = 2*len(pcm) - 2 - j
		}
		if j >= 0 && j < len(pcm) {
			x[i] = pcm[j]
		}
	}
	frames := 1 + (len(x)-melFFT)/melHop - 1
	if frames < 1 {
		frames = 1
	}
	out := make([]float32, melBins*frames)
	filters := melFilter()
	pow := make([]float64, 201)
	max := float32(math.Inf(-1))
	for t := 0; t < frames; t++ {
		for k := 0; k <= 200; k++ {
			var re, im float64
			for n := 0; n < melFFT; n++ {
				v := float64(x[t*melHop+n]) * (.5 - .5*math.Cos(2*math.Pi*float64(n)/melFFT))
				a := -2 * math.Pi * float64(k*n) / melFFT
				re += v * math.Cos(a)
				im += v * math.Sin(a)
			}
			pow[k] = re*re + im*im
		}
		for m := 0; m < melBins; m++ {
			var v float64
			for k := 0; k <= 200; k++ {
				v += filters[m][k] * pow[k]
			}
			z := float32(math.Log10(math.Max(v, 1e-10)))
			out[m*frames+t] = z
			if z > max {
				max = z
			}
		}
	}
	for i, v := range out {
		if v < max-8 {
			v = max - 8
		}
		out[i] = (v + 4) / 4
	}
	return out
}
