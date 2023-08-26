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

static __forceinline__ __device__ float3 poler2xyzDirection(const float theta, const float phi) {
	return make_float3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
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
	return make_float3(
		v.x * t.x + v.y * n.x + v.z * b.x,
		v.x * t.y + v.y * n.y + v.z * b.y,
		v.x * t.z + v.y * n.z + v.z * b.z
	);
}

static __forceinline__ __device__ float3 transform_position(const Matrix4x3& mat, const float3& pos) {
	float4 p = make_float4(pos, 1.0);
	return make_float3(dot(mat.r0, p), dot(mat.r1, p), dot(mat.r2, p));
}

static __forceinline__ __device__ float3 transform_normal(const Matrix4x3& mat, const float3& nor) {
	float4 n = make_float4(nor, 0.0);

	//transposed matrix
	float4 r0 = make_float4(mat.r0.x, mat.r1.x, mat.r2.x, 0.0);
	float4 r1 = make_float4(mat.r0.y, mat.r1.y, mat.r2.y, 0.0);
	float4 r2 = make_float4(mat.r0.z, mat.r1.z, mat.r2.z, 0.0);

	return make_float3(dot(r0, n), dot(r1, n), dot(r2, n));
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

static __forceinline__ __device__ float absdot(const float3& a, const float3& b) {
	return fabsf(dot(a, b));
}

static __forceinline__ __device__ float lerp(const float& a, const float& b, const float& t) {
	return (1 - t) * a + t * b;
}

static __forceinline__ __device__ float smoothstep(const float a, const float b, const float t) {
	float x = clamp((t - a) / (b - a), 0.0f, 1.0f);
	return x * x * (3.0f - 2.0f * x);
}

static __forceinline__ __device__ float step(const float a, const float x) {
	return float(a < x);
}

static __forceinline __device__ float3 hemisphereVector(const float theta, const float phi) {
	return make_float3(cosf(phi) * sinf(theta), cosf(theta), sinf(phi) * sinf(theta));
}
