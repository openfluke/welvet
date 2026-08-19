# lucy — Lucy measuring harness

Mid-stream adaptation metrics for any host (tide, test41-w, live_gpt, a new
sweep). Pure math: no datasets, no train loops, no architectures.

Lucy Score is **live-fit**: can the net **learn while it still serves**.

```
Score        = Throughput × Availability × Acc / 10_000
Availability = InferMs / (InferMs + TrainMs) × 100
Acc          = hard argmax accuracy (AvgAccuracy)
SoftAcc      = serve-confidence — not in Score
ZeroDowntime = Acc × Availability / 100
Realtime     = Throughput × Availability / 100
```

SGD that blocks inference tanks Availability. SoftAcc stays diagnostic.

## Density / synthetic organism (`BuildLPD`)

Given a slice of `Sample` (Score, Acc, Thru, Avail, RAM), lucy ranks
**consciousness** then **memory density** so a new host does not copy tide:

| Symbol | Formula | Why |
|--------|---------|-----|
| Q | geomean(RelAcc, RelThru, RelAvail) vs **learner** peaks | Live-fit vs cells that actually learn |
| LPD | `Q × shrink` vs Acc-champ RAM; **0** if RelAcc &lt; 70% | Condense without the Score/MiB trap |
| Gold | all 3 pillars ≥80% and RAM ≤20% of Acc champ | Trifecta in a small box |
| Trap | RAM ≤20% of Acc champ and Acc keep &lt;70% | Binary / chance Acc looking dense |

Consciousness radar = `row.Consciousness()` → Acc/Thru/Avail keep `[0,1]`.
Memory density radar = `row.MemoryDensity()` → same × shrink (traps at origin).

```go
import "github.com/openfluke/welvet/lucy"

a := lucy.SoftAccOne(pred, target)           // scale 0.10
p := lucy.SoftAccProb(probTrue, 1.0)         // scale 1.0
lucy.Finalize(&snap, lucy.Options{AdaptWindows: 10})

board := lucy.BuildLPD([]lucy.Sample{
    {ID: "f32", Acc: 90, Thru: 200, Avail: 40, Score: 100, RAMKiB: 1000},
    {ID: "int8", Acc: 82, Thru: 180, Avail: 38, Score: 85, RAMKiB: 180},
})
_ = board.Top[0].LPD
```

`tide/metrics` re-exports pulse math. `tide/report.BuildLPD` pretty-prints cell
IDs then calls this package. Dash / PDF stay in tide.

Tests live in `welvet/w2a/tests/lucy` (not under this package).
