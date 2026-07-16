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
