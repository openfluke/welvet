package qwentts

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// bpeTokenizer is a GPT-2 byte-level BPE built from vocab.json + merges.txt
// (the Qwen text tokenizer). Only text->id encoding is implemented; the TTS
// path wraps encoded text with fixed special-token ids.
type bpeTokenizer struct {
	vocab      map[string]int
	mergeRank  map[string]int // "a b" -> rank
	byteToRune map[byte]rune
	runeToByte map[rune]byte
	pat        *regexp.Regexp
}

// gpt2 pre-tokenizer pattern (RE2-compatible subset used across welvet).
var gpt2Pattern = regexp.MustCompile(`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`)

// bytesToUnicode is the canonical GPT-2 reversible byte<->unicode table.
func bytesToUnicode() (map[byte]rune, map[rune]byte) {
	var bs []int
	for i := int('!'); i <= int('~'); i++ {
		bs = append(bs, i)
	}
	for i := int('¡'); i <= int('¬'); i++ {
		bs = append(bs, i)
	}
	for i := int('®'); i <= int('ÿ'); i++ {
		bs = append(bs, i)
	}
	cs := append([]int(nil), bs...)
	n := 0
	for b := 0; b < 256; b++ {
		found := false
		for _, v := range bs {
			if v == b {
				found = true
				break
			}
		}
		if !found {
			bs = append(bs, b)
			cs = append(cs, 256+n)
			n++
		}
	}
	b2r := make(map[byte]rune, 256)
	r2b := make(map[rune]byte, 256)
	for i := range bs {
		b2r[byte(bs[i])] = rune(cs[i])
		r2b[rune(cs[i])] = byte(bs[i])
	}
	return b2r, r2b
}

// loadBPETokenizer reads vocab.json + merges.txt from snapshotDir.
func loadBPETokenizer(snapshotDir string) (*bpeTokenizer, error) {
	vocabPath := filepath.Join(snapshotDir, "vocab.json")
	mergesPath := filepath.Join(snapshotDir, "merges.txt")
	vdata, err := os.ReadFile(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("qwentts tokenizer: %w", err)
	}
	var vocab map[string]int
	if err := json.Unmarshal(vdata, &vocab); err != nil {
		return nil, fmt.Errorf("qwentts vocab.json: %w", err)
	}
	mf, err := os.Open(mergesPath)
	if err != nil {
		return nil, fmt.Errorf("qwentts tokenizer: %w", err)
	}
	defer mf.Close()
	mergeRank := make(map[string]int)
	sc := bufio.NewScanner(mf)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	rank := 0
	for sc.Scan() {
		line := strings.TrimRight(sc.Text(), "\r\n")
		if line == "" || strings.HasPrefix(line, "#version") {
			continue
		}
		parts := strings.SplitN(line, " ", 2)
		if len(parts) != 2 {
			continue
		}
		mergeRank[parts[0]+" "+parts[1]] = rank
		rank++
	}
	b2r, r2b := bytesToUnicode()
	return &bpeTokenizer{
		vocab:      vocab,
		mergeRank:  mergeRank,
		byteToRune: b2r,
		runeToByte: r2b,
		pat:        gpt2Pattern,
	}, nil
}

// encode tokenizes plain text into vocab ids (no special tokens added).
func (t *bpeTokenizer) encode(text string) []int {
	var ids []int
	for _, piece := range t.pat.FindAllString(text, -1) {
		// byte-level mapping
		var sb strings.Builder
		for _, b := range []byte(piece) {
			sb.WriteRune(t.byteToRune[b])
		}
		for _, tok := range t.bpe(sb.String()) {
			if id, ok := t.vocab[tok]; ok {
				ids = append(ids, id)
			}
		}
	}
	return ids
}

// bpe applies merge rules to a byte-mapped string, returning subword tokens.
func (t *bpeTokenizer) bpe(word string) []string {
	syms := make([]string, 0, len(word))
	for _, r := range word {
		syms = append(syms, string(r))
	}
	if len(syms) < 2 {
		return syms
	}
	for {
		bestRank := int(^uint(0) >> 1)
		bestIdx := -1
		for i := 0; i < len(syms)-1; i++ {
			if r, ok := t.mergeRank[syms[i]+" "+syms[i+1]]; ok && r < bestRank {
				bestRank = r
				bestIdx = i
			}
		}
		if bestIdx < 0 {
			break
		}
		merged := syms[bestIdx] + syms[bestIdx+1]
		next := make([]string, 0, len(syms)-1)
		next = append(next, syms[:bestIdx]...)
		next = append(next, merged)
		next = append(next, syms[bestIdx+2:]...)
		syms = next
		if len(syms) < 2 {
			break
		}
	}
	return syms
}
