package wav2vec2

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// Vocab is the CTC character vocabulary (id → token).
type Vocab struct {
	IDToToken []string
	TokenToID map[string]int
	BlankID   int
}

// LoadVocabJSON reads HF vocab.json ({"token": id, ...}).
func LoadVocabJSON(path string, blankID int) (*Vocab, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return LoadVocabBytes(b, blankID)
}

// LoadVocabBytes parses vocab.json bytes.
func LoadVocabBytes(b []byte, blankID int) (*Vocab, error) {
	var raw map[string]int
	if err := json.Unmarshal(b, &raw); err != nil {
		return nil, fmt.Errorf("wav2vec2: vocab: %w", err)
	}
	maxID := -1
	for _, id := range raw {
		if id > maxID {
			maxID = id
		}
	}
	idTo := make([]string, maxID+1)
	for tok, id := range raw {
		if id < 0 || id > maxID {
			return nil, fmt.Errorf("wav2vec2: bad vocab id %d", id)
		}
		idTo[id] = tok
	}
	return &Vocab{IDToToken: idTo, TokenToID: raw, BlankID: blankID}, nil
}

// DecodeCTCGreedy collapses argmax CTC paths: drop blank, collapse repeats, '|' → space.
func (v *Vocab) DecodeCTCGreedy(ids []int) string {
	if v == nil {
		return ""
	}
	var b strings.Builder
	prev := -1
	for _, id := range ids {
		if id == v.BlankID || id == prev {
			prev = id
			continue
		}
		prev = id
		if id < 0 || id >= len(v.IDToToken) {
			continue
		}
		tok := v.IDToToken[id]
		switch tok {
		case "<pad>", "<s>", "</s>", "<unk>":
			continue
		case "|":
			b.WriteByte(' ')
		default:
			b.WriteString(tok)
		}
	}
	return strings.TrimSpace(b.String())
}
