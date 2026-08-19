package parallel

import "strings"

// ShortTrainModeLegend is the display key for compact train-mode names.
const ShortTrainModeLegend = "[T]=Tween  [S]=Split  [FP]=FastProxy  [L]=Linear  [HP]=HeadProxy"

// ShortTrainMode is display-only. Persistence / ParseTrainMode still use String().
func ShortTrainMode(name string) string {
	s := strings.TrimSpace(name)
	if s == "" {
		return s
	}
	repls := [][2]string{
		{"FastProxy", "[FP]"},
		{"HeadProxy", "[HP]"},
		{"Linear", "[L]"},
		{"Split", "[S]"},
		{"Tween", "[T]"},
	}
	for _, r := range repls {
		s = strings.ReplaceAll(s, r[0], r[1])
	}
	return s
}

// Short is the compact title for this mode.
func (m TrainMode) Short() string {
	return ShortTrainMode(m.String())
}
