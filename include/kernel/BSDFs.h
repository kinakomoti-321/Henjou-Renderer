#pragma once

#include <HenjouRenderer/henjouRenderer.h>
#include <cuda/helpers.h>
#include <common/constant.h>
#include <kernel/cmj.h>
#include <kernel/math.h>
#include <kernel/Payload.h>

class Lambert {
private:
	float3 basecolor;
public:
	__device__ Lambert() : basecolor({ 1.0,1.0,1.0 }) {}
	__device__ Lambert(float3 basecolor) : basecolor(basecolor) {}

	__device__ float3 evaluateBSDF(float3 wo, float3 wi) {
		return basecolor / M_PIf;
	}

	__device__ float3 sampleBSDF(const float3& wo, float3& wi, float& pdf, CMJState& state) {
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

	__device__ float3 sampleBSDF(const float3& wo, float3& wi, float& pdf, CMJState& state) {
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

	__device__ float3 sampleBSDF(const float3& wo, float3& wi, float& pdf, CMJState& state) {
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

	__device__ float3 sampleBSDF(const float3& wo, float3& wi, float& pdf, CMJState& state) {
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

	__device__ float3 sampleBSDF(const float3& wo, float3& wi, float& pdf, CMJState& state) {
		if (metallic < 0.5) {
			return lam.sampleBSDF(wo, wi, pdf, state);
		}
		else {
			return ggx.sampleBSDF(wo, wi, pdf, state);	
		}
	}

	__device__ float getPDF(const float3& wo, const float3& wi) {
		return ggx.getPDF(wo, wi);
	}
};
