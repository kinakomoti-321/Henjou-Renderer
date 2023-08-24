#pragma once
#include <optix.h>

#include <HenjouRenderer/henjouRenderer.h>
#include <cuda/helpers.h>
#include <sutil/vec_math.h>

extern "C" {
	__constant__ Params params;
}

struct Payload
{
	bool is_hit = false;

	float3 ray_origin;
	float3 ray_direction;

	float3 position = { 0.0,0.0,0.0 };
	float3 normal = { 0.0,0.0,0.0 };
	float3 vert_color = { 0.0,0.0,0.0 };
	float2 texcoord = { 0.0,0.0 };
	int material_id = 0;

	float3 basecolor = { 0.0,0.0,0.0 };
	float metallic = 0.0;
	float roughness = 0.0;
	float sheen = 0.0;
	float clearcoat = 0.0;
	float ior = 1.0;
	float transmission = 1.0;

	float3 emission = { 0.0,0.0,0.0 };
	bool is_light = false;

	int primitive_id = 0;
	int instance_id = 0;
};

static __forceinline__ __device__ void* unpack_ptr(unsigned int i0,
	unsigned int i1)
{
	const unsigned long long uptr =
		static_cast<unsigned long long>(i0) << 32 | i1;
	void* ptr = reinterpret_cast<void*>(uptr);
	return ptr;
}

static __forceinline__ __device__ void pack_ptr(void* ptr, unsigned int& i0,
	unsigned int& i1)
{
	const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
	i0 = uptr >> 32;
	i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ Payload* get_payload_ptr()
{
	const unsigned int u0 = optixGetPayload_0();
	const unsigned int u1 = optixGetPayload_1();
	return reinterpret_cast<Payload*>(unpack_ptr(u0, u1));
}
