#pragma once

#include <sutil/sutil.h>
#include <common/matrix.h>

// 4x3 matrix
// Affine
// r0.x r0.y r0.z r0.w
// r1.x r1.y r1.z r1.w
// r2.x r2.y r2.z r2.w
// 0    0    0    1
struct Matrix4x3 {
	float4 r0;
	float4 r1;
	float4 r2;
};
