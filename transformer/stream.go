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
	inThink      bool
	// HideThink drops <think>…</think> spans from the streamed view (thinking off).
	HideThink bool
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
	if s.HideThink {
		diff = filterThinkDelta(diff, &s.inThink)
	} else {
		if strings.Contains(diff, "<think>") {
			s.inThink = true
		}
		if strings.Contains(diff, "</think>") {
			s.inThink = false
		}
	}
	if diff == "" {
		s.lastLen = len(full)
		return
	}
	if !silent {
		fmt.Print(diff)
	}
	if callback != nil {
		callback(diff)
	}
	s.sb.WriteString(diff)
	s.lastLen = len(full)
}

// ForceCloseThink emits </think> when the model got stuck inside a think block.
func (s *Streamer) ForceCloseThink(silent bool, callback func(string)) {
	if s == nil || !s.inThink {
		return
	}
	piece := "</think>\n\n"
	s.inThink = false
	if !silent {
		fmt.Print(piece)
	}
	if callback != nil {
		callback(piece)
	}
	s.sb.WriteString(piece)
}

// InThink reports whether the stream is currently inside a <think> block.
func (s *Streamer) InThink() bool { return s != nil && s.inThink }

// String returns the streamed assistant text (trimmed).
func (s *Streamer) String() string {
	return strings.TrimSpace(s.sb.String())
}

// filterThinkDelta hides Qwen3 <think>…</think> spans while streaming.
func filterThinkDelta(diff string, inThink *bool) string {
	if diff == "" {
		return ""
	}
	var out strings.Builder
	i := 0
	for i < len(diff) {
		if !*inThink {
			if j := strings.Index(diff[i:], "<think>"); j >= 0 {
				out.WriteString(diff[i : i+j])
				*inThink = true
				i += j + len("<think>")
				continue
			}
			out.WriteString(diff[i:])
			break
		}
		if j := strings.Index(diff[i:], "</think>"); j >= 0 {
			*inThink = false
			i += j + len("</think>")
			continue
		}
		break // still inside think — drop rest of this chunk
	}
	return out.String()
}
