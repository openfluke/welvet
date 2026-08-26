#include "textflag.h"

// double dotF64AccF64Avx2(double *x, double *w, int n, double prev)
// AVX2+FMA — dual 4-wide float64 accumulators (8 elems/iter), Plan 9 Dense WireF64.
TEXT ·dotF64AccF64Avx2(SB), NOSPLIT, $0-40
	MOVQ    x+0(FP), AX
	MOVQ    w+8(FP), BX
	MOVQ    n+16(FP), CX
	MOVSD   prev+24(FP), X12

	VXORPD  Y0, Y0, Y0
	VXORPD  Y1, Y1, Y1

	CMPQ    CX, $8
	JL      reduce4

loop8:
	VMOVUPD (AX), Y4
	VMOVUPD 32(AX), Y5
	VMOVUPD (BX), Y6
	VMOVUPD 32(BX), Y7
	VFMADD231PD Y6, Y4, Y0
	VFMADD231PD Y7, Y5, Y1
	ADDQ    $64, AX
	ADDQ    $64, BX
	SUBQ    $8, CX
	CMPQ    CX, $8
	JGE     loop8

reduce4:
	VADDPD  Y1, Y0, Y0
	CMPQ    CX, $4
	JL      reduce

loop4:
	VMOVUPD (AX), Y4
	VMOVUPD (BX), Y5
	VFMADD231PD Y5, Y4, Y0
	ADDQ    $32, AX
	ADDQ    $32, BX
	SUBQ    $4, CX
	CMPQ    CX, $4
	JGE     loop4

reduce:
	VEXTRACTF128 $1, Y0, X1
	VADDPD  X0, X1, X0
	VPERMILPD $1, X0, X1
	VADDSD  X1, X0, X0
	VADDSD  X12, X0, X0

	TESTQ   CX, CX
	JE      done

tail:
	MOVSD   (AX), X4
	MOVSD   (BX), X5
	VMULSD  X5, X4, X4
	VADDSD  X4, X0, X0
	ADDQ    $8, AX
	ADDQ    $8, BX
	DECQ    CX
	JNZ     tail

done:
	VZEROUPPER
	MOVSD   X0, ret+32(FP)
	RET
