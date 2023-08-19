#pragma once

#include <optix.h>

#include <HenjouRenderer/henjouRenderer.h>
#include <cuda/helpers.h>

#include <sutil/vec_math.h>
#include <kernel/cmj.h>
#include <kernel/math.h>

extern "C" {
	__constant__ Params params;
}

struct Payload
{
	bool is_hit = false;

	float3 position = { 0.0,0.0,0.0 };
	float3 normal = { 0.0,0.0,0.0 };
	float3 vert_color = { 0.0,0.0,0.0 };
	float2 texcoord = { 0.0,0.0 };
	int material_id = 0;

	float3 basecolor = { 0.0,0.0,0.0};
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

static __forceinline__ __device__ void computeRay(uint3 idx, uint3 dim, float3& origin, float3& direction)
{
	const float3 U = params.cam_u;
	const float3 V = params.cam_v;
	const float3 W = params.cam_w;
	const float2 d = 2.0f * make_float2(
		static_cast<float>(idx.x) / static_cast<float>(dim.x),
		static_cast<float>(idx.y) / static_cast<float>(dim.y)
	) - 1.0f;

	origin = params.cam_eye;
	direction = normalize(d.x * U + d.y * V + W);
}

static __forceinline__ __device__ void TraceOcculution(
	OptixTraversableHandle handle,
	float3 ray_origin,
	float3 ray_direction,
	float tmin,
	float tmax,
	Payload* prd
) {
	unsigned int u0, u1;
	pack_ptr(prd, u0, u1);
	optixTrace(
		params.traversal_handle,
		ray_origin,
		ray_direction,
		tmin,                // Min intersection distance
		tmax,               // Max intersection distance
		0.0f,                // rayTime -- used for motion blur
		OptixVisibilityMask(255), // Specify always visible
		OPTIX_RAY_FLAG_NONE,
		1,                   // SBT offset   -- See SBT discussion
		2,                   // SBT stride   -- See SBT discussion
		0,                   // missSBTIndex -- See SBT discussion
		u0,
		u1
	);

}

static __forceinline__ __device__ void RayTrace(
	OptixTraversableHandle handle,
	float3 ray_origin,
	float3 ray_direction,
	float tmin,
	float tmax,
	Payload* prd
) {
	unsigned int u0, u1;
	pack_ptr(prd, u0, u1);
	optixTrace(
		params.traversal_handle,
		ray_origin,
		ray_direction,
		tmin,                // Min intersection distance
		tmax,               // Max intersection distance
		0.0f,                // rayTime -- used for motion blur
		OptixVisibilityMask(255), // Specify always visible
		OPTIX_RAY_FLAG_NONE,
		0,                   // SBT offset   -- See SBT discussion
		2,                   // SBT stride   -- See SBT discussion
		0,                   // missSBTIndex -- See SBT discussion
		u0,
		u1
	);

}


__device__ struct Ray {
	float3 origin;
	float3 direction;
	float tmin = 0.001f;
	float tmax = 1e16f;
};

class Lambert {
private:
	float3 basecolor;
public:
	__device__ Lambert() : basecolor({ 1.0,1.0,1.0 }) {}
	__device__ Lambert(float3 basecolor) : basecolor(basecolor) {}

};

__device__ float3 Pathtracing(float3 firstRayOrigin, float3 firstRayDirection, CMJstate& state) {
	float3 LTE = { 0.0,0.0,0.0 };
	float3 throughput = { 1.0,1.0,1.0 };
	float russian_p = 1.0;
	int MaxDepth = 10;

	Ray ray;
	ray.origin = firstRayOrigin;
	ray.direction = firstRayDirection;

	for (int i = 0; i < MaxDepth; i++) {
		russian_p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));

		if (russian_p < cmj_1d(state)) {
			break;	
		}
		
		throughput /= russian_p;
		Payload prd;
		RayTrace(
			params.traversal_handle,
			ray.origin,
			ray.direction,
			0.0001f,                // Min intersection distance
			1e16f,               // Max intersection distance
			&prd
		);

		if (!prd.is_hit) {
			LTE += throughput * prd.emission;
			break;
		}

		if (prd.is_light) {
			LTE += throughput * prd.emission;
			break;
		}

		float3 t, b, n;
		n = prd.normal;
		orthonormal_basis(n, t, b);
		float pdf = 1.0;
		float3 local_wo = world_to_local(-ray.direction, t, n, b);
		float2 xi = cmj_2d(state);
		float3 local_wi = cosineSampling(xi.x,xi.y, pdf);
		float3 wi = local_to_world(local_wi, t, n, b);

		float3 bsdf = prd.basecolor / M_PIf;
		throughput *= bsdf * fabs(dot(wi, n)) / pdf;

		ray.origin = prd.position;
		ray.direction = wi;
	}

	return LTE;
}
