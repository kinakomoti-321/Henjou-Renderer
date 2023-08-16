#pragma once

#include <cmath>
#include <sutil/sutil.h>

float3 make_float3(const float2& v, float z) {
	return make_float3(v.x, v.y, z);
}

float3 make_float3(float x, const float2& v) {
	return make_float3(x, v.x, v.y);
}

float4 make_float4(const float3& v, float w) {
	return make_float4(v.x, v.y, v.z, w);
}

float4 make_float4(float x, const float3& v) {
	return make_float4(x, v.x, v.y, v.z);
}

float4 make_float4(const float2& a, const float2& b) {
	return make_float4(a.x, a.y, b.x, b.y);
}

float3 operator-(const float3& a) {
	return make_float3(-a.x, -a.y, -a.z);
}

float3 operator+(const float3& a, const float3& b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

float3 operator-(const float3& a, const float3& b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

float3 operator*(const float3& a, const float3& b) {
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

float3 operator/(const float3& a, const float3& b) {
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

float3 operator*(float a, const float3& b) {
	return make_float3(a * b.x, a * b.y, a * b.z);
}

float3 operator*(const float3& a, float b) {
	return make_float3(a.x * b, a.y * b, a.z * b);
}

float3 operator/(const float3& a, float b) {
	float inv = 1.0f / b;
	return a * inv;
}

float3 operator/(float a, const float3& b) {
	return make_float3(a / b.x, a / b.y, a / b.z);
}

float norm(const float3& a) {
	return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);	
}

float norm2(const float3& a) {
	return a.x * a.x + a.y * a.y + a.z * a.z;
}

float3 normalize(const float3& a) {
	float invLen = 1.0f / sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
	return a * invLen;
}

float3 cross(const float3& a, const float3& b) {
	return make_float3(a.y * b.z - a.z * b.y,
				a.z * b.x - a.x * b.z,
				a.x * b.y - a.y * b.x);
}

float dot(const float3& a, const float3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
