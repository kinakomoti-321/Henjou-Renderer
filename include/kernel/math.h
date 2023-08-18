#pragma once

#include <cuda/helpers.h>
#include <sutil/vec_math.h>

__device__ float3 cosineSampling(float u, float v, float& pdf) {
	float phi = 2 * M_PIf * v;
	float theta = 0.5 * acos(1 - 2.0 * u);
	float cosTheta = cos(theta);
	float sinTheta = sin(theta);
	pdf = cosTheta / M_PIf;
	return make_float3(cos(phi) * sinTheta, cosTheta, sin(phi) * sinTheta);

}

__device__ float3 hemisphereSampling(float u, float v, float& pdf) {
	float phi = 2 * M_PIf * v;
	float theta = acos(u);
	float cosTheta = cos(theta);
	float sinTheta = sin(theta);
	pdf = 1.0 / (2.0 * M_PIf);
	return make_float3(cos(phi) * sinTheta, cosTheta, sin(phi) * sinTheta);
}

static __forceinline__ __device__ void orthonormal_basis(const float3& normal, float3& tangent, float3& binormal)
{
	float sign = copysignf(1.0f, normal.z);
	const float a = -1.0f / (sign + normal.z);
	const float b = normal.x * normal.y * a;
	tangent = make_float3(1.0f + sign * normal.x * normal.x * a, sign * b,
		-sign * normal.x);
	binormal = make_float3(b, sign + normal.y * normal.y * a, -normal.y);
}

static __forceinline__ __device__ float3 world_to_local(const float3& v,
	const float3& t,
	const float3& n,
	const float3& b)
{
	return make_float3(dot(v, t), dot(v, n), dot(v, b));
}

static __forceinline__ __device__ float3 local_to_world(const float3& v,
	const float3& t,
	const float3& n,
	const float3& b)
{
	return make_float3(v.x * t.x + v.y * n.x + v.z * b.x,
		v.x * t.y + v.y * n.y + v.z * b.y,
		v.x * t.z + v.y * n.z + v.z * b.z);
}
