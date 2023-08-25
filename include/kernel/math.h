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

__device__ float3 shlickFresnel(const float3& F0, const float3& w, const float3& n) {
	float term1 = 1.0 - dot(w, n);
	return (1.0 - F0) * pow(term1, 5.0) + F0;
}

__device__ float shlickFresnel(const float no, const float ni, const float3& w, const float3& n) {
	float F0 = (no - ni) / (no + ni);
	F0 = F0 * F0;

	float term1 = 1.0 - dot(w, n);
	return F0 + (1.0 - F0) * pow(term1, 5.0);
}

static __forceinline__ __device__ float3 poler2xyzDirection(const float theta,const float phi) {
	return make_float3(sin(theta) * cos(phi), cos(theta),  sin(theta) * sin(phi));
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

static __forceinline__ __device__ float norm2(const float3& v) {
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

static __forceinline__ __device__  bool refract(const float3& v, const float3& n, float ior1, float ior2,
	float3& r) {
	const float3 t_h = -ior1 / ior2 * (v - dot(v, n) * n);

	// ‘S”½ŽË
	if (norm2(t_h) > 1.0) return false;

	const float3 t_p = -sqrtf(fmaxf(1.0f - norm2(t_h), 0.0f)) * n;
	r = t_h + t_p;

	return true;
}

static __forceinline__ __device__ float absdot(const float3& a,const float3& b) {
	return fabsf(dot(a, b));
}

static __forceinline __device__ float lerp(const float& a, const float& b, const float& t) {
	return (1 - t) * a + t * b;
}

