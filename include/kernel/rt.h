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

static __forceinline__ __device__ void computeRay(uint3 idx, uint3 dim, float3& origin, float3& direction)
{
	const float3 V = params.camera_dir;
	float3 U;
	float3 W;
	orthonormal_basis(V, U, W);
	const float2 d = 2.0f * make_float2(
		static_cast<float>(idx.x) / static_cast<float>(dim.x),
		static_cast<float>(idx.y) / static_cast<float>(dim.y)
	) - 1.0f;

	origin = params.camera_pos;
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

	__device__ float3 evaluateBSDF(float3 wo, float3 wi) {
		return basecolor / M_PIf;
	}

	__device__ float3 sampleBSDF(const float3& wo, float3& wi, float& pdf, CMJstate& state) {
		float2 xi = cmj_2d(state);
		wi = cosineSampling(xi.x, xi.y, pdf);
		return basecolor / M_PIf;
	}

	__device__ float getPDF(const float3& wo, const float3& wi) {
		return  abs(wi.y) / M_PIf;
	}
};

class GGX {
private:
	float3 F0;
	float alpha;

	__device__ float GGX_D(const float3& wm) {
		float term1 = wm.x * wm.x / (alpha * alpha) + wm.z * wm.z / (alpha * alpha) + wm.y * wm.y;
		float term2 = PI * alpha * alpha * term1 * term1;
		return 1.0 / term2;
	}

	__device__ float GGX_G2_HeightCorrelated(const float3& wi, const float3& wo) {
		return 1.0 / (1.0 + GGX_Lambda(wi) + GGX_Lambda(wo));
	}

	__device__ float GGX_Lambda(const float3& w) {
		float term1 = (alpha * alpha * w.x * w.x + alpha * alpha * w.z * w.z) / (w.y * w.y);
		float term2 = sqrt(1.0 + term1);
		return 0.5 * (term2 - 1.0);
	}

	__device__ float3 sampleD(float2 uv) {
		float theta = atan(alpha * sqrt(uv.x) / sqrt(1.0 - uv.x));
		float phi = PI2 * uv.y;
		return poler2xyzDirection(theta, phi);
	}

public:
	__device__ GGX() {
		F0 = { 0.04,0.04,0.04 };
		alpha = 0.5;
	}
	__device__ GGX(float3 iF0, float iroughness) {
		F0 = iF0;
		alpha = clamp(iroughness * iroughness, 0.0001f, 1.0f);
	}

	__device__ float3 evaluateBSDF(float3 wo, float3 wi) {
		const float3 wm = normalize(wo + wi);

		float ggxD = GGX_D(wm);
		float ggxG2 = GGX_G2_HeightCorrelated(wi, wo);
		float3 ggxF = shlickFresnel(F0, wi, wm);

		return ggxD * ggxG2 * ggxF / (4.0 * wo.y * wi.y);
	}

	__device__ float3 sampleBSDF(const float3& wo, float3& wi, float& pdf, CMJstate& state) {
		float2 xi = cmj_2d(state);
		const float3 wm = sampleD(xi);

		wi = reflect(-wo, wm);

		if (wi.y <= 0.0) {
			pdf = 1.0;
			return { 0.0,0.0,0.0 };
		}

		float ggxD = GGX_D(wm);
		float ggxG2 = GGX_G2_HeightCorrelated(wi, wo);
		float3 ggxF = shlickFresnel(F0, wi, wm);

		pdf = ggxD * wm.y / (4.0 * dot(wo, wm));

		return ggxD * ggxG2 * ggxF / (4.0 * wo.y * wi.y);
	}

	__device__ float getPDF(const float3& wo, const float3& wi) {

	}

};


static __forceinline__ __device__ float norm2(const float3& v) {
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

static __forceinline__ __device__ float3 Reflect(const float3& v, const float3& n) {
	return v - 2.0 * dot(v, n) * n;
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

static __forceinline__ __device__  float fresnel(const float3& w, const float3& n, float ior1, float ior2) {
	float f0 = (ior1 - ior2) / (ior1 + ior2);
	f0 = f0 * f0;
	float delta = fmaxf(1.0f - dot(w, n), 0.0f);
	return f0 + (1.0f - f0) * delta * delta * delta * delta * delta;
}

class IdealGlass {
private:
	float3 rho;
	float ior;

public:
	__device__ IdealGlass() {
		rho = make_float3(0);
		ior = 1.0;
	}

	__device__ IdealGlass(const float3& rho, const float& ior) :rho(rho), ior(ior) {}

	__device__ float3 sampleBSDF(const float3& wo, float3& wi, float& pdf, CMJstate& state) {
		float ior_o, ior_i;
		float3 n;

		float3 lwo = wo;
		float3 lwi = make_float3(0.0);

		ior_o = 1.0;
		ior_i = ior;

		float sign = 1.0;

		n = make_float3(0, 1, 0);

		if (wo.y < 0.0) {
			ior_o = ior;
			ior_i = 1.0;
			lwo.y = -lwo.y;
			sign = -1.0;
		}

		const float fr = fresnel(lwo, n, ior_o, ior_i);

		float3 evalbsdf;

		float p = cmj_1d(state);

		if (p < fr) {
			lwi = Reflect(-lwo, n);
			pdf = fr;
			evalbsdf = fr * rho / fabsf(lwi.y);
		}
		else {
			float3 t;
			if (refract(lwo, n, ior_o, ior_i, t)) {
				lwi = t;
				pdf = 1;
				evalbsdf = (1.0 - fr) * rho / fabsf(lwi.y);
			}
			else {
				lwi = Reflect(-lwo, n);
				pdf = 1;
				evalbsdf = rho / fabsf(lwi.y);
			}
		}

		wi = lwi;
		wi.y = sign * wi.y;

		return evalbsdf;
	}

	__device__ float3 evalueateBSDF(const float3& wo, const float3& wi) {
		return make_float3(0);
	}

	__device__ float pdfBSDF(const float3& wo, const float3& wi) {
		return 0;
	}

};

class DietricGGX {
private:
	float ior;
	float alpha;

	__device__ float GGX_D(const float3& wm) {
		float term1 = wm.x * wm.x / (alpha * alpha) + wm.z * wm.z / (alpha * alpha) + wm.y * wm.y;
		float term2 = PI * alpha * alpha * term1 * term1;
		return 1.0 / term2;
	}

	__device__ float GGX_G1(const float3 w) {
		return 1.0 / (1.0 + GGX_Lambda(w));
	}

	__device__ float GGX_G2_HeightCorrelated(const float3& wi, const float3& wo) {
		return 1.0 / (1.0 + GGX_Lambda(wi) + GGX_Lambda(wo));
	}

	__device__ float GGX_Lambda(const float3& w) {
		float term1 = (alpha * alpha * w.x * w.x + alpha * alpha * w.z * w.z) / (w.y * w.y);
		float term2 = sqrt(1.0 + term1);
		return 0.5 * (term2 - 1.0);
	}


	__device__ float3 sampleD(float2 uv) {
		float theta = atan(alpha * sqrt(uv.x) / sqrt(1.0 - uv.x));
		float phi = PI2 * uv.y;
		return poler2xyzDirection(theta, phi);
	}

public:
	__device__ DietricGGX() {
		ior = 1.0;
		alpha = 0.5;
	}

	__device__ DietricGGX(float ior_i, float roughness_i) {
		ior = ior_i;
		alpha = (roughness_i * roughness_i);
	}

	__device__ float3 evaluateBSDF(float3 wo, float3 wi) {
	}

	__device__ float3 sampleBSDF(const float3& wo, float3& wi, float& pdf, CMJstate& state) {
		float ior_o, ior_i;
		float3 n;

		float3 lwo = wo;
		float3 lwi = make_float3(0.0);

		ior_o = 1.0;
		ior_i = ior;

		float sign = 1.0;

		n = make_float3(0, 1, 0);

		if (wo.y < 0.0) {
			ior_o = ior;
			ior_i = 1.0;
			lwo.y = -lwo.y;
			sign = -1.0;
		}

		const float fr = fresnel(lwo, n, ior_o, ior_i);

		float3 evalbsdf;

		float p = cmj_1d(state);

		if (p < fr) {
			lwi = Reflect(-lwo, n);
			pdf = fr;
			evalbsdf = make_float3(fr)/ fabsf(lwi.y);
		}
		else {
			float3 t;
			if (refract(lwo, n, ior_o, ior_i, t)) {
				lwi = t;
				pdf = 1;
				evalbsdf = (1.0 - make_float3(fr)) / fabsf(lwi.y);
			}
			else {
				lwi = Reflect(-lwo, n);
				pdf = 1;
				evalbsdf = make_float3(fr) / fabsf(lwi.y);
			}
		}

		wi = lwi;
		wi.y = sign * wi.y;

		return evalbsdf;
	}


};
class BSDF {
private:
	float3 basecolor;
	float roughness;
	float metallic;
	float sheen;
	float clearcoat;
	float ior;
	float transmission;

	Lambert lam;
	GGX ggx;
	DietricGGX glass;
	IdealGlass idealglass;
public:
	__device__ BSDF() {
		basecolor = { 1.0,1.0,1.0 };
		roughness = 1.0;
		metallic = 1.0;
		sheen = 0.0;
		clearcoat = 0.0;
		ior = 1.0;
		transmission = 0.0;
	}

	__device__ BSDF(const Payload& pyload) {
		basecolor = pyload.basecolor;
		roughness = pyload.roughness;
		metallic = pyload.metallic;
		sheen = pyload.sheen;
		clearcoat = pyload.clearcoat;
		ior = pyload.ior;
		transmission = pyload.transmission;

		lam = Lambert(basecolor);
		ggx = GGX(basecolor, roughness);
		ior = 1.5;
		glass = DietricGGX(ior, roughness);
		idealglass = IdealGlass(basecolor, ior);
	}

	__device__ float3 evaluateBSDF(float3 wo, float3 wi) {
		return ggx.evaluateBSDF(wo, wi);
	}

	__device__ float3 sampleBSDF(const float3& wo, float3& wi, float& pdf, CMJstate& state) {
		if (metallic < 0.5) {
			return lam.sampleBSDF(wo, wi, pdf, state);
		}
		else {
			pdf = 1.0;
			//return glass.sampleBSDF(wo, wi, pdf, state);
			return idealglass.sampleBSDF(wo, wi, pdf, state);
			//return glass.sampleBSDF(wo, wi, pdf, state);
			//return ggx.sampleBSDF(wo, wi, pdf, state);
		}
	}

	__device__ float getPDF(const float3& wo, const float3& wi) {
		return ggx.getPDF(wo, wi);
	}
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

		float3 local_wi;

		float3 bsdf = surface_bsdf.sampleBSDF(local_wo, local_wi, pdf, state);

		float3 wi = local_to_world(local_wi, t, n, b);

		throughput *= bsdf * fabs(dot(wi, n)) / pdf;

		ray.origin = prd.position;
		ray.direction = wi;
	}

	return LTE;
}
