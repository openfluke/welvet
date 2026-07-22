//go:build amd64

#include "textflag.h"

// void saxpyF32Avx2(float *y, float alpha, float *x, int n)
// y[i] += alpha * x[i]
TEXT ·saxpyF32Avx2(SB), NOSPLIT, $0-28
	MOVQ    y+0(FP), DI
	MOVSS   alpha+8(FP), X15
	VBROADCASTSS X15, Y15
	MOVQ    x+16(FP), SI
	MOVQ    n+24(FP), CX

	CMPQ    CX, $8
	JL      tail

loop8:
	VMOVUPS (DI), Y0
	VMOVUPS (SI), Y1
	VFMADD231PS Y1, Y15, Y0
	VMOVUPS Y0, (DI)
	ADDQ    $32, DI
	ADDQ    $32, SI
	SUBQ    $8, CX
	CMPQ    CX, $8
	JGE     loop8

tail:
	CMPQ    CX, $0
	JE      done

tailLoop:
	MOVSS   (SI), X0
	MULSS   X15, X0
	ADDSS   (DI), X0
	MOVSS   X0, (DI)
	ADDQ    $4, DI
	ADDQ    $4, SI
	DECQ    CX
	JNE     tailLoop

done:
	VZEROUPPER
	RET
