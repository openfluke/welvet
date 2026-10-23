package nativeqnn

// PhaseTiming is the last HTP graph phase split (nanoseconds).
type PhaseTiming struct {
	CreateNs   int64
	FinalizeNs int64
	ExecuteNs  int64
}

func (p PhaseTiming) CreateUS() float64   { return float64(p.CreateNs) / 1e3 }
func (p PhaseTiming) FinalizeUS() float64 { return float64(p.FinalizeNs) / 1e3 }
func (p PhaseTiming) ExecuteUS() float64  { return float64(p.ExecuteNs) / 1e3 }
func (p PhaseTiming) PrepareUS() float64  { return p.CreateUS() + p.FinalizeUS() }
