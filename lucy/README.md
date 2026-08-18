# lucy — Lucy measuring harness

Mid-stream adaptation metrics shared by test41-w, tide, and live_mnist.

Lucy Score is **live-fit**: can the net **learn while it still serves**.

```
Score        = Throughput × Availability × Acc / 10_000
Availability = InferMs / (InferMs + TrainMs) × 100
Acc          = hard argmax accuracy (AvgAccuracy)
SoftAcc      = serve-confidence (softmax vs true class) — not in Score
ZeroDowntime = Acc × Availability / 100
```

SGD that blocks inference tanks Availability. Proxy / FastProxy / MeshTween
paths that train without stalling the live loop keep Score. SoftAcc stays as a
diagnostic; it is not the Acc pillar.

**In this package:** SoftAcc, Availability, AdaptPct, Score, ZeroDowntime,
Stability, Consistency, `Window` / `Snapshot`, `Finalize`.

**Not in this package:** datasets (sine/MNIST), train loops, architectures.

```go
import "github.com/openfluke/welvet/lucy"

a := lucy.SoftAccOne(pred, target)           // scale 0.10
p := lucy.SoftAccProb(probTrue, 1.0)         // scale 1.0
lucy.Finalize(&snap, lucy.Options{AdaptWindows: 10})
```

`tide/metrics` re-exports this package for existing tide callers.

Tests live in `welvet/w2a/tests/lucy` (not under this package).
