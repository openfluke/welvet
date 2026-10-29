// CUDA integration supports explicit sessions and optional routed execution
// on Linux/Windows amd64.
// EnableRouting connects JSON/Exec.Backend CUDA placements to native sessions;
// close the returned Router after the workload. Routed updates return to Go
// automatically. Explicit sessions use the following ownership boundary:
//
//	session, err := nativecuda.Mount(model)
//	// session.TrainMSE(input, target, parallel.ModeNormalBP, learningRate)
//	// session.SyncToGo()  // before Go ENTITY save or Go-side execution
//	// session.Close()
//
// Build with -tags welvet_cuda for a CUDA-only application: this disables the
// independent nativecpp cgo binding, avoiding a CPU sibling library dependency.
// Other Go CPU reference implementations remain available for explicit use.
// Mount returns an error for unsupported shapes/configurations; it never falls
// back to CPU training. See native/cuda/README.md for artifact deployment.
package nativecuda
