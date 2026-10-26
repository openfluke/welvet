// Run without the welvet_metal build tag to keep the C++ binding enabled.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/model/entity"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/nativemetal"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/placement"
	"log"
	"os"
	"strconv"
)

func main() {
	configPath := flag.String("config", "", "optional execution JSON file")
	save := flag.String("save", "", "optional trained Go ENTITY output file")
	flag.Parse()
	if !nativecpp.NativeCPUAvailable() {
		log.Fatal("C++ binding unavailable: use the normal build with its native artifact, without -tags welvet_metal")
	}
	r, err := nativemetal.EnableRouting()
	if err != nil {
		log.Fatal(err)
	}
	defer r.Close()
	layers := map[string]nativecpp.LayerBackend{}
	var ops []any
	for i := 0; i < 13; i++ {
		l, e := dense.NewConfigured(2, 2, core.ActivationTanh, core.DTypeFloat32, quant.FormatNone, []float32{.8, .12, -.15, .9})
		if e != nil {
			log.Fatal(e)
		}
		ops = append(ops, l)
		backend := "go_cpu"
		if i >= 4 && i <= 8 {
			backend = "native_cpu"
		}
		if i >= 9 {
			backend = "metal"
		}
		layers[strconv.Itoa(i)] = nativecpp.LayerBackend{Backend: backend}
	}
	cfg := nativecpp.ExecConfig{Default: "go_cpu", Volumes: map[string]nativecpp.VolumeBackend{"main": {Sectors: map[string]nativecpp.SectorBackend{"0,0,0,0": {Layers: layers}}}}}
	if *configPath != "" {
		raw, e := os.ReadFile(*configPath)
		if e != nil {
			log.Fatal(e)
		}
		if e = json.Unmarshal(raw, &cfg); e != nil {
			log.Fatal(e)
		}
	}
	model, err := parallel.NewStack(ops...)
	if err != nil {
		log.Fatal(err)
	}
	if err = placement.Stack(model, cfg, "main", "0,0,0,0"); err != nil {
		log.Fatal(err)
	}
	x := &core.Tensor[float32]{Shape: []int{1, 2}, Data: []float32{.5, -.3}}
	target := core.NewTensor[float32](1, 2)
	var loss float64
	for i := 0; i < 5; i++ {
		loss, err = parallel.TrainStackMSE(model, x, target, parallel.ModeNormalBP, .03)
		if err != nil {
			log.Fatal(err)
		}
	}
	fmt.Printf("Go layers 0–3 → C++ layers 4–8 → Metal layers 9–12 (default config)\nloss=%g; native Metal dispatches=%+v\n", loss, r.Stats())
	if *save != "" {
		if err = entity.WriteCameralFile(*save, model, parallel.ModeNormalBP); err != nil {
			log.Fatal(err)
		}
		fmt.Println("saved", *save)
	}
}
