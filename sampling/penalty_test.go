package sampling

import "testing"

func TestHasRepeatedNGram(t *testing.T) {
	toks := []uint32{1, 2, 3, 4, 5, 3, 4, 5}
	if !HasRepeatedNGram(toks, 3) {
		t.Fatal("expected 3-gram repeat")
	}
	if HasRepeatedNGram(toks, 4) {
		t.Fatal("unexpected 4-gram repeat")
	}
}

func TestApplyRepetitionPenalty(t *testing.T) {
	logits := []float32{0, 2, -2, 1}
	ApplyRepetitionPenalty(logits, []uint32{1, 2}, 1.5, 64)
	if logits[1] >= 2 {
		t.Fatalf("positive logit not penalized: %v", logits[1])
	}
	if logits[2] >= -2 {
		t.Fatalf("negative logit not penalized: %v", logits[2])
	}
}
