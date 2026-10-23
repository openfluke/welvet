package nativeqnn

import "github.com/openfluke/welvet/core"

func init() {
	core.SetQNNAvailableChecker(Available)
}
