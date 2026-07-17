package mosstts

import (
	"fmt"
	"os"
	"path/filepath"

	sentencepiece "github.com/eliben/go-sentencepiece"
)

// SentencePiece wraps a BPE SentencePiece processor (tokenizer.model).
type SentencePiece struct {
	proc *sentencepiece.Processor
}

// LoadSentencePiece loads tokenizer.model (or resolves it next to vocab_pieces.json).
func LoadSentencePiece(path string) (*SentencePiece, error) {
	if filepath.Ext(path) == ".json" {
		cand := filepath.Join(filepath.Dir(path), "tokenizer.model")
		if st, err := os.Stat(cand); err == nil && !st.IsDir() {
			path = cand
		} else {
			return nil, fmt.Errorf("mosstts: need tokenizer.model next to %s", path)
		}
	}
	return LoadTokenizerModel(path)
}

// LoadTokenizerModel loads a SentencePiece .model protobuf.
func LoadTokenizerModel(modelPath string) (*SentencePiece, error) {
	proc, err := sentencepiece.NewProcessorFromPath(modelPath)
	if err != nil {
		return nil, fmt.Errorf("mosstts sentencepiece: %w", err)
	}
	return &SentencePiece{proc: proc}, nil
}

// Encode matches Python SentencePieceProcessor.encode(text).
func (sp *SentencePiece) Encode(text string) []int {
	if sp == nil || sp.proc == nil {
		return nil
	}
	toks := sp.proc.Encode(text)
	ids := make([]int, len(toks))
	for i, t := range toks {
		ids[i] = t.ID
	}
	return ids
}

// EncodeRaw aliases Encode.
func (sp *SentencePiece) EncodeRaw(text string) []int { return sp.Encode(text) }

// Piece stub.
func (sp *SentencePiece) Piece(id int) string { return "" }
