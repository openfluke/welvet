# lucy — Lucy measuring harness

Mid-stream adaptation metrics shared by test41-w, tide, and live_mnist.

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
