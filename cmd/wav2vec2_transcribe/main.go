package main

import (
	"fmt"
	"os"
	"time"

	"github.com/openfluke/welvet/model/wav2vec2"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintf(os.Stderr, "usage: %s <hf-model-dir> <audio.wav>\n", os.Args[0])
		os.Exit(2)
	}
	dir, wav := os.Args[1], os.Args[2]
	fmt.Println("loading", dir, "…")
	t0 := time.Now()
	m, err := wav2vec2.LoadHFDir(dir)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("loaded in", time.Since(t0).Round(time.Millisecond))
	t1 := time.Now()
	text, err := m.TranscribeFile(wav)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("transcribed in", time.Since(t1).Round(time.Millisecond))
	fmt.Println(text)
}
