# Go to welvet.cpp dispatch

`core.BackendCPUTiled` runs the Go implementation. `core.BackendNativeCPU`
selects the independent C++ CPU backend through the C ABI. It does not select
Metal or CUDA. The Apple backend uses `core.BackendMetal` / `nativemetal`;
NVIDIA uses `core.BackendCuda` / `nativecuda`.

The FormatNone baseline now connects native forward and incoming-gradient
backward for CNN1/2/3, ConvT1/2/3, Embedding, RNN, LSTM, MHA, Mamba, KMeans,
Sequential/Residual Dense layouts, and Float32 GDN. These replace dispatchers
that previously called Go host implementations despite requesting NativeCPU.
Dense, SwiGLU, RMSNorm, LayerNorm, Softmax and configured Parallel already had
native C ABI paths. Stack/heterogeneous containers retain Go graph orchestration;
Metacognition retains live Go rule handling around native Dense math.

Backward returns dInput/dWeights without mutating native parameters. Gradient
shapes follow the Go layer contract, including matrix-shaped Embedding/CNN and
row-shaped recurrent packs. Existing Go SGD/credit schedulers can consume them.
C++ whole-model train bindings retain their supported Dense layout contracts;
selecting NativeCPU does not imply that all host scheduling, SGD or proxy-credit
arithmetic moves into the native library.

Unsupported packed/advanced configurations retain their existing host behavior.
MHA cache/custom mask/QK-norm/cross-attention/dropout/custom RoPE configurations,
heterogeneous Sequential/Residual layouts and stateful GDN are not newly claimed
as native whole-layer baselines. Those host paths may still call native child
projections. The raw C ABI's model reconstruction can re-encode low-bit weights;
CPU/GPU and native roundtrips are compared using the project's numerical bands.

To verify execution rather than a requested backend label:

```go
before := nativecpp.CompletedCallsFor("mamba_forward")
_, post, err := mamba.Forward(layer, input)
// Check err, then nativecpp.CompletedCallsFor("mamba_forward") > before.
```

Counters are monotonic and process-wide. `CompletedCalls()` counts all successful
C ABI calls; `CompletedCallsFor(name)` identifies the actual checked operation,
for example `mamba_normalbp`, `embedding_forward` or `cnn2_normalbp`.

Tests live in w2a:

```sh
cd /Users/openfluke/gs/w2a
go test ./tests/go_v_metal -run '^(TestGoNativeCPPRouting|TestGoNativeMetalRouting.*)$' -count=1 -v
```

The audit exercises 23 baseline families across the supported 34 native parameter
dtypes (GDN Float32), checks actual native operations, and compares forward,
incoming gradients and SGD results with explicitly selected Go CPU references.
