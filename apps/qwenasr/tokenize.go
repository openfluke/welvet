package qwenasr

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

type bpeTokenizer struct {
	vocab map[string]int
	rev   map[int]string
	rank  map[string]int
	b2r   map[byte]rune
	r2b   map[rune]byte
	pat   *regexp.Regexp
}

var bpePat = regexp.MustCompile(`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`)

func byteMaps() (map[byte]rune, map[rune]byte) {
	bs := []int{}
	for i := 33; i <= 126; i++ {
		bs = append(bs, i)
	}
	for i := 161; i <= 172; i++ {
		bs = append(bs, i)
	}
	for i := 174; i <= 255; i++ {
		bs = append(bs, i)
	}
	cs := append([]int(nil), bs...)
	n := 0
	for b := 0; b < 256; b++ {
		ok := false
		for _, x := range bs {
			if x == b {
				ok = true
			}
		}
		if !ok {
			bs = append(bs, b)
			cs = append(cs, 256+n)
			n++
		}
	}
	a := map[byte]rune{}
	z := map[rune]byte{}
	for i, b := range bs {
		a[byte(b)] = rune(cs[i])
		z[rune(cs[i])] = byte(b)
	}
	return a, z
}
func loadTokenizer(dir string) (*bpeTokenizer, error) {
	b, e := os.ReadFile(filepath.Join(dir, "vocab.json"))
	if e != nil {
		return nil, e
	}
	v := map[string]int{}
	if e = json.Unmarshal(b, &v); e != nil {
		return nil, e
	}
	t := &bpeTokenizer{vocab: v, rev: map[int]string{}, rank: map[string]int{}, pat: bpePat}
	for k, id := range v {
		t.rev[id] = k
	}
	t.b2r, t.r2b = byteMaps()
	f, e := os.Open(filepath.Join(dir, "merges.txt"))
	if e != nil {
		return nil, e
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	i := 0
	for sc.Scan() {
		p := strings.SplitN(sc.Text(), " ", 2)
		if len(p) == 2 && !strings.HasPrefix(p[0], "#") {
			t.rank[p[0]+" "+p[1]] = i
			i++
		}
	}
	return t, sc.Err()
}
func (t *bpeTokenizer) bpe(s string) []string {
	x := []string{}
	for _, r := range s {
		x = append(x, string(r))
	}
	for len(x) > 1 {
		bi, br := -1, int(^uint(0)>>1)
		for i := 0; i < len(x)-1; i++ {
			if r, ok := t.rank[x[i]+" "+x[i+1]]; ok && r < br {
				bi, br = i, r
			}
		}
		if bi < 0 {
			break
		}
		x = append(append(append([]string{}, x[:bi]...), x[bi]+x[bi+1]), x[bi+2:]...)
	}
	return x
}
func (t *bpeTokenizer) encode(s string) []int {
	var out []int
	for _, p := range t.pat.FindAllString(s, -1) {
		var b strings.Builder
		for _, x := range []byte(p) {
			b.WriteRune(t.b2r[x])
		}
		for _, z := range t.bpe(b.String()) {
			if id, ok := t.vocab[z]; ok {
				out = append(out, id)
			}
		}
	}
	return out
}
func (t *bpeTokenizer) decode(ids []int) string {
	var b []byte
	var text strings.Builder
	flush := func() {
		if len(b) == 0 {
			return
		}
		text.Write(b)
		b = b[:0]
	}
	for _, id := range ids {
		if id == asrTextID {
			flush()
			text.WriteString("<asr_text>")
			continue
		}
		if id >= eosID {
			continue
		}
		tok, ok := t.rev[id]
		if !ok {
			continue
		}
		for _, r := range tok {
			if x, ok := t.r2b[r]; ok {
				b = append(b, x)
			}
		}
	}
	flush()
	return text.String()
}
