package swiglu

import "github.com/openfluke/welvet/nativecpp"

// ApplyExecConfig applies the shared JSON/native per-layer placement.
func (l *Layer) ApplyExecConfig(cfg nativecpp.ExecConfig, volume, sector, layer string) {
	if l != nil {
		nativecpp.ApplyExecConfig(cfg, volume, sector, layer, &l.Exec)
	}
}
