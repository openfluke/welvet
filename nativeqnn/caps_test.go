package nativeqnn_test

import (
	"runtime"
	"testing"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/nativeqnn"
)

func TestCapsAndSyncGate(t *testing.T) {
	if !nativeqnn.Available() {
		if err := nativeqnn.RequireLayer("dense", "float32"); err == nil {
			t.Fatal("expected RequireLayer error when unavailable")
		}
		l, err := dense.New(2, 2, core.ActivationLinear, core.DTypeFloat32)
		if err != nil {
			t.Fatal(err)
		}
		if err := l.SyncToQNN(); err == nil {
			t.Fatal("expected SyncToQNN fail without DLL")
		}
		t.Log("stub platform: caps gate OK")
		return
	}
	if runtime.GOOS != "windows" || runtime.GOARCH != "arm64" {
		t.Fatalf("Available() true on unexpected %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	layers := nativeqnn.SupportedLayers()
	if len(layers) == 0 {
		t.Fatal("SupportedLayers empty")
	}
	t.Logf("version=%s layers=%v", nativeqnn.Version(), layers)
	if !nativeqnn.LayerSupported("dense", "float32") {
		t.Fatal("dense/float32 should be supported")
	}
	if nativeqnn.LayerSupported("kmeans", "float32") {
		t.Fatal("kmeans should not be supported")
	}
	l, err := dense.New(4, 4, core.ActivationLinear, core.DTypeFloat32)
	if err != nil {
		t.Fatal(err)
	}
	if err := l.SyncToQNN(); err != nil {
		t.Fatal(err)
	}
	if !l.QNNSynced() {
		t.Fatal("expected QNNSynced")
	}
	l.Exec.Backend = core.BackendQNN
	x := core.NewTensor[float32](1, 4)
	for i := range x.Data {
		x.Data[i] = float32(i) * 0.1
	}
	_, _, err = dense.Forward(l, x)
	if err != nil {
		t.Logf("ForwardQNN (needs HTP/HVCI): %v", err)
	} else {
		t.Log("ForwardQNN OK")
	}
	if err := l.SyncFromQNN(); err != nil {
		t.Fatal(err)
	}
}
