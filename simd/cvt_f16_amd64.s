#include "textflag.h"

// void cvtF16x8Avx(uint16 *src, float32 *dst)
// F16C: 8×IEEE half → 8×float32 (true HW convert, then DotTile does AVX2 FMA).
TEXT ·cvtF16x8Avx(SB), NOSPLIT, $0-16
	MOVQ src+0(FP), AX
	MOVQ dst+8(FP), BX
	VCVTPH2PS (AX), Y0
	VMOVUPS Y0, (BX)
	VZEROUPPER
	RET

// void cvtBF16x8Avx(uint16 *src, float32 *dst)
// 8×bfloat16 → 8×float32 via zero-extend + shift-left 16 (AVX2).
TEXT ·cvtBF16x8Avx(SB), NOSPLIT, $0-16
	MOVQ src+0(FP), AX
	MOVQ dst+8(FP), BX
	VMOVDQU (AX), X0
	VPMOVZXWD X0, Y0
	VPSLLD $16, Y0, Y0
	VMOVUPS Y0, (BX)
	VZEROUPPER
	RET
