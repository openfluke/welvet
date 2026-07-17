//go:build amd64

#include "textflag.h"

// float32 binaryWordSum1Avx2(float32 *x, uint32 word)
// Sum x[j] for each bit j set in word (j = 0..31). Exact float32.
TEXT ·binaryWordSum1Avx2(SB), NOSPLIT, $0-20
	MOVQ	x+0(FP), SI
	MOVL	word+8(FP), BX
	VXORPS	X0, X0, X0		// acc (4 lanes)

	// 8 iterations × 4 floats
	MOVQ	$8, CX

loop4:
	VMOVUPS	(SI), X1

	// Build 4-lane mask from low 4 bits of BX: lane i = 0xFFFFFFFF if bit i set.
	MOVL	BX, AX
	ANDL	$1, AX
	NEGL	AX			// 0 or 0xFFFFFFFF
	MOVL	AX, X2
	SHRL	$1, BX

	MOVL	BX, AX
	ANDL	$1, AX
	NEGL	AX
	PINSRD	$1, AX, X2
	SHRL	$1, BX

	MOVL	BX, AX
	ANDL	$1, AX
	NEGL	AX
	PINSRD	$2, AX, X2
	SHRL	$1, BX

	MOVL	BX, AX
	ANDL	$1, AX
	NEGL	AX
	PINSRD	$3, AX, X2
	SHRL	$1, BX

	VPAND	X2, X1, X1
	VADDPS	X1, X0, X0

	ADDQ	$16, SI
	DECQ	CX
	JNE	loop4

	// Horizontal sum of X0 → scalar
	VHADDPS	X0, X0, X0
	VHADDPS	X0, X0, X0
	MOVSS	X0, ret+16(FP)
	RET
