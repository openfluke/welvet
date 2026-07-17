package sentencepiece

import (
	"strings"
	"unicode"
)

const whitespaceSeparator = "▁"

func normalize(text string, addDummyPrefix, removeExtraWhitespaces bool) string {
	if removeExtraWhitespaces {
		text = collapseAndTrimWhitespace(text)
	}
	// Empty input stays empty (no dummy ▁). Matches C++/Python Encode("\n")→[].
	if text == "" {
		return ""
	}
	if addDummyPrefix && text[0] != ' ' {
		text = " " + text
	}
	return replaceSpacesBySeparator(text)
}

// collapseAndTrimWhitespace mirrors SentencePiece remove_extra_whitespaces:
// all unicode whitespace → single ' ', then trim.
func collapseAndTrimWhitespace(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	prevSpace := false
	for _, r := range s {
		if unicode.IsSpace(r) {
			if prevSpace {
				continue
			}
			b.WriteByte(' ')
			prevSpace = true
			continue
		}
		prevSpace = false
		b.WriteRune(r)
	}
	return strings.TrimSpace(b.String())
}

func replaceSpacesBySeparator(text string) string {
	return strings.ReplaceAll(text, " ", whitespaceSeparator)
}

func replaceSeparatorsBySpace(text string) string {
	return strings.ReplaceAll(text, whitespaceSeparator, " ")
}
