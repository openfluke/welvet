// A complete mount → Metal train → explicit pull → Go ENTITY save example.
package main

import (
	"flag"
	"fmt"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/model/entity"
	"github.com/openfluke/welvet/nativemetal"
	"github.com/openfluke/welvet/quant"
	"log"
)

func main() {
	library := flag.String("library", "", "Metal dylib path (default: environment, executable directory, installed Go artifact)")
	save := flag.String("save", "", "optional output ENTITY path")
	flag.Parse()
	layer, e := dense.NewConfigured(2, 2, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, []float32{.3, -.2, .1, .4})
	if e != nil {
		log.Fatal(e)
	}
	model, e := parallel.NewStack(layer)
	if e != nil {
		log.Fatal(e)
	}
	s, e := nativemetal.Mount(model, nativemetal.Options{LibraryPath: *library})
	if e != nil {
		log.Fatal(e)
	}
	defer s.Close()
	input := core.NewTensor[float32](1, 2)
	copy(input.Data, []float32{.5, .25})
	target := core.NewTensor[float32](1, 2)
	var loss float64
	for i := 0; i < 5; i++ {
		loss, e = s.TrainMSE(input, target, parallel.ModeNormalBP, .1)
		if e != nil {
			log.Fatal(e)
		}
	}
	if e = s.SyncToGo(); e != nil {
		log.Fatal(e)
	}
	if *save != "" {
		if e = entity.WriteCameralFile(*save, model, parallel.ModeNormalBP); e != nil {
			log.Fatal(e)
		}
	}
	fmt.Printf("welvet.metal %s\n%s\n5 training steps; loss=%g; weights synchronized to Go\n", s.Version(), s.Identity(), loss)
	if *save != "" {
		fmt.Println("saved:", *save)
	}
}
