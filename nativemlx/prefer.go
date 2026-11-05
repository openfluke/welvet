package nativemlx

import "github.com/openfluke/welvet/core"

func init() {
	core.SetMLXAvailableChecker(Available)
}
