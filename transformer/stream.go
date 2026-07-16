package transformer

import (
	"fmt"
	"strings"
)

// Streamer prints incremental decoded text as new tokens arrive (poly parity).
type Streamer struct {
	Decode       func(tokens []uint32) string
	lastLen      int
	promptLenRaw int
	sb           strings.Builder
	replacer     *strings.Replacer
}

// NewStreamer creates a stream decoder. promptTokens is the encoded prompt prefix.
func NewStreamer(decode func(tokens []uint32) string, promptTokens []uint32) *Streamer {
	promptTextRaw := decode(promptTokens)
	imEnd := "<|" + "im_end" + "|>"
	return &Streamer{
		Decode:       decode,
		lastLen:      len(promptTextRaw),
		promptLenRaw: len(promptTextRaw),
		replacer: strings.NewReplacer(
			imEnd, "",
			"<|im_start|>assistant", "",
			"<|im_start|>user", "",
			"<|im_start|>system", "",
			"<|im_start|>assistant\n", "",
		),
	}
}

// Push decodes allTokens and prints/callbacks only the new suffix since last push.
func (s *Streamer) Push(allTokens []uint32, silent bool, callback func(string)) {
	full := s.Decode(allTokens)
	if len(full) <= s.lastLen {
		return
	}
	diff := full[s.lastLen:]
	diff = s.replacer.Replace(diff)
	if !silent {
		fmt.Print(diff)
	}
	if callback != nil {
		callback(diff)
	}
	s.sb.WriteString(diff)
	s.lastLen = len(full)
}

// String returns the streamed assistant text (trimmed).
func (s *Streamer) String() string {
	return strings.TrimSpace(s.sb.String())
}
