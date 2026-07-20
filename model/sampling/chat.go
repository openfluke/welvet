package sampling

import "strings"

// ReplyLooksDegenerate detects token-salad / mojibake tails that small models
// often enter when a turn runs too long without an EOS.
func ReplyLooksDegenerate(s string) bool {
	s = strings.TrimSpace(s)
	if len(s) < 48 {
		return false
	}
	runes := []rune(s)
	win := runes
	if len(win) > 96 {
		win = win[len(win)-96:]
	}
	bad := 0
	letters := 0
	spaces := 0
	weirdCase := 0
	prevUpper := false
	for _, r := range win {
		switch {
		case r == '\uFFFD' || r == 'ï' || r == 'â' || r == 'Â' || r == 'Ã' ||
			r == '¼' || r == '½' || r == '¬' || r == 'ð' || r == 'Ð' || r == 'ÿ':
			bad++
		case r == ' ' || r == '\n' || r == '\t':
			spaces++
			prevUpper = false
		case (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z'):
			letters++
			up := r >= 'A' && r <= 'Z'
			if up && prevUpper {
				weirdCase++
			}
			prevUpper = up
		default:
			prevUpper = false
		}
	}
	if bad >= 2 {
		return true
	}
	if weirdCase >= 6 && letters >= 30 {
		return true
	}
	if len(win) >= 48 && spaces < 2 && letters >= 24 {
		return true
	}
	if len(win) >= 72 && spaces < 4 && letters >= 36 {
		return true
	}
	return false
}

// SanitizeChatReply cuts a reply at the last clean sentence before degeneration.
// If nothing clean remains, returns empty (caller should drop the turn from history).
func SanitizeChatReply(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	if !ReplyLooksDegenerate(s) {
		return trimToLastSentence(s)
	}
	runes := []rune(s)
	best := ""
	for i := 1; i <= len(runes); i++ {
		pref := string(runes[:i])
		if ReplyLooksDegenerate(pref) {
			break
		}
		trimmed := strings.TrimSpace(pref)
		if endsSentence(trimmed) {
			best = trimmed
		} else if best == "" && len(trimmed) >= 24 {
			best = trimmed
		}
	}
	if best == "" {
		return ""
	}
	return trimToLastSentence(best)
}

func endsSentence(s string) bool {
	if s == "" {
		return false
	}
	switch s[len(s)-1] {
	case '.', '!', '?', '\n', '"', '\'':
		return true
	default:
		return false
	}
}

func trimToLastSentence(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	last := -1
	runes := []rune(s)
	for i, r := range runes {
		if r == '.' || r == '!' || r == '?' {
			last = i
		}
	}
	if last >= 0 && last+1 < len(runes) {
		rest := strings.TrimSpace(string(runes[last+1:]))
		if len(rest) > 0 && len(rest) < 40 && !strings.ContainsAny(rest, ".!?") {
			return strings.TrimSpace(string(runes[:last+1]))
		}
	}
	return s
}
