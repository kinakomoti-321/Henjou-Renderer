#pragma once

#include <optix.h>

#include <HenjouRenderer/henjouRenderer.h>
#include <cuda/helpers.h>

#include <sutil/vec_math.h>
#include <kernel/Payload.h>
#include <kernel/cmj.h>
#include <kernel/math.h>
#include <kernel/BSDFs.h>
#include <kernel/light_sample.h>

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


__device__ float3 Pathtrace(float3 firstRayOrigin, float3 firstRayDirection, CMJState state) {
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
			0.001f,                // Min intersection distance
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

		BSDF surface_bsdf(prd);

		float3 t, b, n;
		n = prd.normal;
		orthonormal_basis(n, t, b);
		float pdf = 1.0;
		float3 local_wo = world_to_local(-ray.direction, t, n, b);

		float3 local_wi = { 0.0,1.0,0.0 };

		float2 xi = cmj_2d(state);
		float3 bsdf;

		{
			bsdf = surface_bsdf.sampleBSDF(local_wo, local_wi, pdf, state);
		}


		float3 wi = local_to_world(local_wi, t, n, b);

		throughput *= bsdf * fabs(dot(wi, n)) / pdf;

		ray.origin = prd.position;
		ray.direction = wi;
	}

	return LTE;
}


__device__ float3 NEE(float3 firstRayOrigin, float3 firstRayDirection, CMJState state) {
	float3 LTE = { 0.0,0.0,0.0 };
	float3 throughput = { 1.0,1.0,1.0 };
	float russian_p = 1.0;
	int MaxDepth = 10;

	Ray ray;
	ray.origin = firstRayOrigin;
	ray.direction = firstRayDirection;

	for (int depth = 0; depth < MaxDepth; depth++) {
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
			0.001f,                // Min intersection distance
			1e16f,               // Max intersection distance
			&prd
		);

		if (!prd.is_hit) {
			if (depth == 0) {
				LTE += throughput * prd.emission;
			}
			break;
		}

		if (prd.is_light) {
			if (depth == 0) {
				LTE += throughput * prd.emission;
			}
			break;
		}

		BSDF surface_bsdf(prd);

		float3 t, b, n;
		n = prd.normal;
		orthonormal_basis(n, t, b);

		float3 local_wo = world_to_local(-ray.direction, t, n, b);

		//NEE
		{
			Payload prd_shadow;
			prd_shadow.is_hit = false;

			float light_pdf;
			float3 light_normal;
			float3 light_position;
			float3 light_emission;

			light_position = light_sample(state, light_pdf, light_normal, light_emission);

			float3 light_direction = light_position - prd.position;
			float light_distance = length(light_direction);
			light_direction = normalize(light_direction);

			TraceOcculution(
				params.traversal_handle,
				prd.position,
				light_direction,
				0.001f,
				light_distance - 0.001f,
				&prd_shadow
			);

			if (!prd_shadow.is_hit) {
				float cosine1 = absdot(n, light_direction);
				float cosine2 = absdot(light_normal, -light_direction);

				float3 local_wi = world_to_local(light_direction, t, n, b);
				float3 bsdf = surface_bsdf.evaluateBSDF(local_wo, local_wi);

				float G = cosine2 / (light_distance * light_distance);
				LTE += throughput * (bsdf * G * cosine1 / light_pdf) * light_emission;
			}
		}

		float pdf = 1.0;

		float3 local_wi = { 0.0,1.0,0.0 };

		float2 xi = cmj_2d(state);
		float3 bsdf;

		bsdf = surface_bsdf.sampleBSDF(local_wo, local_wi, pdf, state);


		float3 wi = local_to_world(local_wi, t, n, b);

		throughput *= bsdf * fabs(dot(wi, n)) / pdf;

		ray.origin = prd.position;
		ray.direction = wi;
	}

	return LTE;
}

__device__ float3 MIS(float3 firstRayOrigin, float3 firstRayDirection, CMJState state) {
	float3 LTE = { 0.0,0.0,0.0 };
	float3 throughput = { 1.0,1.0,1.0 };
	float russian_p = 1.0;
	int MaxDepth = 10;

	Ray ray;
	ray.origin = firstRayOrigin;
	ray.direction = firstRayDirection;

	for (int depth = 0; depth < MaxDepth; depth++) {
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
			0.001f,                // Min intersection distance
			1e16f,               // Max intersection distance
			&prd
		);

		if (!prd.is_hit) {
			if (depth == 0) {
				LTE += throughput * prd.emission;
			}
			break;
		}

		if (prd.is_light) {
			if (depth == 0) {
				LTE += throughput * prd.emission;
			}
			break;
		}

		BSDF surface_bsdf(prd);

		float3 t, b, n;
		n = prd.normal;
		orthonormal_basis(n, t, b);

		float3 local_wo = world_to_local(-ray.direction, t, n, b);

		//NEE
		{
			Payload prd_shadow;
			prd_shadow.is_hit = false;

			float light_pdf;
			float3 light_normal;
			float3 light_position;
			float3 light_emission;

			light_position = light_sample(state, light_pdf, light_normal, light_emission);

			float3 light_direction = light_position - prd.position;
			float light_distance = length(light_direction);
			light_direction = normalize(light_direction);

			TraceOcculution(
				params.traversal_handle,
				prd.position,
				light_direction,
				0.001f,
				light_distance - 0.001f,
				&prd_shadow
			);

			if (!prd_shadow.is_hit) {
				float cosine1 = absdot(n, light_direction);
				float cosine2 = absdot(light_normal, -light_direction);

				float3 local_wi = world_to_local(light_direction, t, n, b);
				float3 bsdf = surface_bsdf.evaluateBSDF(local_wo, local_wi);

				float G = cosine2 / (light_distance * light_distance);
				
				float pt_pdf = surface_bsdf.getPDF(local_wo,local_wi) * G;
				float mis_weight = light_pdf / (light_pdf + pt_pdf);


				LTE += throughput * (bsdf * G * cosine1 / light_pdf) * mis_weight * light_emission;
			}
		}

		//Pathtrace
		{
			float pt_pdf = 1.0;
			float3 local_wi = { 0.0,1.0,0.0 };
			float2 xi = cmj_2d(state);
			float3 bsdf;
			bsdf = surface_bsdf.sampleBSDF(local_wo, local_wi, pt_pdf, state);

			float3 wi = local_to_world(local_wi, t, n, b);
			
			float cosine1 = absdot(wi, n);

			Payload prd_next;
			prd_next.is_hit = false;
			prd_next.is_light = false;
			
			RayTrace(
				params.traversal_handle,
				prd.position,
				wi,
				0.001f,                // Min intersection distance
				1e16f,               // Max intersection distance
				&prd_next
			);

			if (prd_next.is_hit && prd_next.is_light) {
				float cosine2 = absdot(-wi,prd_next.normal);
				float light_distance = length(prd_next.position - prd.position);

				float invG = light_distance * light_distance / cosine2;

				float nee_pdf = getLightPDF(prd_next.primitive_id, prd_next.instance_id) * invG;
				float mis_weight = (pt_pdf) / (pt_pdf + nee_pdf);

				LTE += throughput * prd_next.emission * mis_weight * bsdf * cosine1 / pt_pdf ;
			}
			if (!prd_next.is_hit) {
				LTE += throughput * prd_next.emission * bsdf * cosine1 / pt_pdf;
			}
		}

		float pdf = 1.0;

		float3 local_wi = { 0.0,1.0,0.0 };

		float2 xi = cmj_2d(state);
		float3 bsdf;

		bsdf = surface_bsdf.sampleBSDF(local_wo, local_wi, pdf, state);


		float3 wi = local_to_world(local_wi, t, n, b);

		throughput *= bsdf * fabs(dot(wi, n)) / pdf;

		ray.origin = prd.position;
		ray.direction = wi;
	}

	return LTE;
}
