#pragma once

#include <HenjouRenderer/henjouRenderer.h>
#include <cuda/helpers.h>
#include <sutil/vec_math.h>

#include <common/constant.h>
#include <kernel/cmj.h>
#include <kernel/math.h>
#include <kernel/Payload.h>
#include <kernel/disneyBRDF.h>

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
		return 1.0f / term2;
	}

	__device__ float GGX_G1(const float3& w) {
		return 1.0f / (1.0f + GGX_Lambda(w));
	}

	__device__ float GGX_G2_HeightCorrelated(const float3& wi, const float3& wo) {
		return 1.0f / (1.0f + GGX_Lambda(wi) + GGX_Lambda(wo));
	}

	__device__ float GGX_Lambda(const float3& w) {
		float term1 = (alpha * alpha * w.x * w.x + alpha * alpha * w.z * w.z) / (w.y * w.y);
		return 0.5f * (-1.0f + sqrtf(term1));
	}

	//https://arxiv.org/pdf/2306.05044.pdf
	__device__ float3 sampleVisibleNormal(float2 uv, float3 wo) {
		float3 strech_wo = normalize(make_float3(wo.x * alpha, wo.y, wo.z * alpha));

		float phi = 2.0f * PI * uv.x;
		float z = fma((1.0f - uv.y), (1.0f + strech_wo.y), -strech_wo.y);
		float sinTheta = sqrtf(clamp(1.0f - z * z, 0.0f, 1.0f));
		float x = sinTheta * cos(phi);
		float y = sinTheta * sin(phi);

		float3 c = make_float3(x, z, y);

		float3 h = c + strech_wo;

		float3 wm = normalize(make_float3(h.x * alpha, h.y, h.z * alpha));

		return wm;
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
		const float3 wm = sampleVisibleNormal(xi,wo);
		//const float3 wm = sampleD(xi);

		wi = reflect(-wo, wm);

		if (wi.y <= 0.0) {
			pdf = 1.0f;
			return { 0.0,0.0,0.0 };
		}

		float ggxD = GGX_D(wm);
		float ggxG2 = GGX_G2_HeightCorrelated(wi, wo);
		float3 ggxF = shlickFresnel(F0, wi, wm);

		float jacobian = 0.25f / absdot(wo, wm);
		//Walter PDF
		//pdf = ggxD * wm.y * jacobian;

		//Visible Normal PDF
		float ggxG1 = GGX_G1(wo);
		pdf = ggxD * ggxG1 * absdot(wo,wm) * jacobian / fabsf(wo.y);

		return ggxD * ggxG2 * ggxF / (4.0 * wo.y * wi.y);
	}

	__device__ float getPDF(const float3& wo, const float3& wi) {

	}

};

class IdealGlass {
private:
	float3 rho_;
	float ior_;

public:
	__device__ IdealGlass() {
		rho_ = make_float3(1);
		ior_ = 1.0;
	}

	__device__ IdealGlass(const float3& rho, const float& ior) :rho_(rho), ior_(ior) {
	}

	__device__ float3 sampleBSDF(const float3& wo, float3& wi, float& pdf, CMJState& state) {
		float ior_o, ior_i;
		float3 n;

		float3 lwo = wo;
		float3 lwi = make_float3(0.0);

		ior_o = 1.0;
		ior_i = ior_;

		float sign = 1.0;

		n = make_float3(0, 1, 0);

		if (wo.y < 0.0) {
			ior_o = ior_;
			ior_i = 1.0;
			lwo.y = -lwo.y;
			sign = -1.0;
		}

		const float fr = shlickFresnel(ior_o, ior_i, lwo, n);

		float3 evalbsdf;

		float p = cmj_1d(state);

		if (p < fr) {
			lwi = reflect(-lwo, n);
			pdf = 1;
			evalbsdf = rho_ / fabsf(lwi.y);
		}
		else {
			float3 t;
			if (refract(lwo, n, ior_o, ior_i, t)) {
				lwi = t;
				pdf = 1;
				evalbsdf = rho_ / fabsf(lwi.y);
			}
			else {
				lwi = reflect(-lwo, n);
				pdf = 1;
				evalbsdf = rho_ / fabsf(lwi.y);
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

class MetaMaterialGlass {
private:
	float3 rho_;
	float ior_;

public:
	__device__ MetaMaterialGlass() {
		rho_ = make_float3(1);
		ior_ = 1.0;
	}

	__device__ MetaMaterialGlass(const float3& rho, const float& ior) :rho_(rho), ior_(ior) {

	}

	__device__ float3 sampleBSDF(const float3& wo, float3& wi, float& pdf, CMJState& state) {
		float ior_o, ior_i;
		float3 n;

		float3 lwo = wo;
		float3 lwi = make_float3(0.0);

		ior_o = 1.0;
		ior_i = ior_;

		float sign = 1.0;

		n = make_float3(0, 1, 0);

		if (wo.y < 0.0) {
			ior_o = ior_;
			ior_i = 1.0;
			lwo.y = -lwo.y;
			sign = -1.0;
		}

		const float fr = shlickFresnel(ior_o, ior_i, lwo, n);

		float3 evalbsdf;

		float p = cmj_1d(state);

		if (p < fr) {
			lwi = reflect(-lwo, n);
			pdf = 1;
			evalbsdf = rho_ / fabsf(lwi.y);
		}
		else {
			float3 t;
			if (refract(lwo, n, ior_o, ior_i, t)) {
				lwi = reflect(-t, make_float3(0, -1, 0));
				pdf = 1;
				evalbsdf = rho_ / fabsf(lwi.y);
			}
			else {
				lwi = reflect(-lwo, n);
				pdf = 1;
				evalbsdf = rho_ / fabsf(lwi.y);
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
	MetaMaterialGlass idealglass;

	DisneyBRDF disney;

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
		metallic = pyload.metallic;
		lam = Lambert(basecolor);
		ggx = GGX(basecolor, pyload.roughness);
		ior = 1.5;
		idealglass = MetaMaterialGlass(make_float3(1.0), ior);
		disney = DisneyBRDF(pyload);
	}

	__device__ float3 evaluateBSDF(float3 wo, float3 wi) {
		return disney.evaluateBSDF(wo, wi);
	}

	__device__ float3 sampleBSDF(const float3& wo, float3& wi, float& pdf, CMJState& state) {
		return disney.sampleBSDF(wo, wi, pdf, state);
	}

	__device__ float getPDF(const float3& wo, const float3& wi) {
		return disney.getPDF(wo, wi);
	}
};
