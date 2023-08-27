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

	__device__ float GGX_Lambda(const float3& v) {
		float delta = 1.0f + (alpha * alpha * v.x * v.x + alpha * alpha * v.z * v.z) / (v.y * v.y);
		return (-1.0 + sqrtf(delta)) / 2.0f;
		//float term1 = (w.x * w.x / (alpha * alpha) + w.z * w.z / (alpha * alpha)) / (w.y * w.y);
		//return 0.5f * (-1.0f + sqrtf(term1));
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

	__device__ float3 evaluateBSDF(const float3& wo,const float3& wi) {
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

//class GGX1 {
//
//public:
//	float3 F0;
//	float alpha;
//
//	__device__ float lambda(const float3& v) const {
//		/*
//		float absTan = BSDFMath::tanTheta(v);
//		if (isinf(absTan)) return 0.0;
//		float delta = fmaxf(alpha * BSDFMath::tanTheta(v), 0.0f);
//		return fmaxf((-1.0f + sqrtf(1.0f + delta * delta)) / 2.0f, 0.0f);
//		*/
//
//		float delta = 1 + (alpha * alpha * v.x * v.x + alpha * alpha * v.z * v.z) / (v.y * v.y);
//		return (-1.0 + sqrtf(delta)) / 2.0f;
//	}
//
//	//Height correlated Smith shadowing-masking
//	__device__ float shadowG(const float3& o, const float3& i) {
//		return 1.0f / (1.0f + lambda(o) + lambda(i));
//	}
//	__device__ float shadowG_1(const float3& v) {
//		return 1.0f / (1.0f + lambda(v));
//	}
//
//	//GGX normal distiribution
//	__device__ float GGX_D(const float3& m) {
//		/*
//		const float tan2theta = BSDFMath::tan2Theta(m);
//		const float cos4theta = BSDFMath::cos2Theta(m) * BSDFMath::cos2Theta(m);
//		const float term = 1.0f + tan2theta / (alpha * alpha);
//		return 1.0f / ((PI * alpha * alpha * cos4theta) * term * term);
//		*/
//
//		float delta = m.x * m.x / (alpha * alpha) + m.z * m.z / (alpha * alpha) + m.y * m.y;
//		return 1.0 / (PI * alpha * alpha * delta * delta);
//	}
//
//	//Importance Sampling
//	//Walter distribution sampling
//	__device__ float3 walterSampling(float u, float v) {
//		float theta = atanf(alpha * sqrtf(u) / sqrtf(fmaxf(1.0f - u, 0.0f)));
//		float phi = 2.0f * PI * v;
//		return hemisphereVector(theta, phi);
//	}
//
//
//public:
//	__device__ GGX1() {
//		F0 = { 0.0,0.0,0.0 };
//		alpha = 0.001f;
//	}
//	__device__ GGX1(const float3& F0, const float& in_alpha) :F0(F0) {
//		alpha = fmaxf(in_alpha * in_alpha, 0.001f);
//	}
//
//	__device__ float3 visibleNormalSampling(const float3& V_, float u, float v) {
//		float a_x = alpha, a_y = alpha;
//		float3 V = normalize(make_float3(a_x * V_.x, V_.y, a_y * V_.z));
//
//		float3 n = make_float3(0, 1, 0);
//		if (V.y > 0.99) n = make_float3(1, 0, 0);
//		float3 T1 = normalize(cross(V, n));
//		float3 T2 = normalize(cross(T1, V));
//
//		float r = sqrtf(u);
//		float a = 1.0f / (1.0f + V.y);
//		float phi;
//		if (a > v) {
//			phi = PI * v / a;
//		}
//		else {
//			phi = PI * (v - a) / (1.0f - a) + PI;
//		}
//
//		float P1 = r * cosf(phi);
//		float P2 = r * sinf(phi);
//		if (a < v) P2 *= V.y;
//
//		float3 N = P1 * T1 + P2 * T2 + sqrtf(fmaxf(1.0f - P1 * P1 - P2 * P2, 0.0f)) * V;
//
//		N = normalize(make_float3(a_x * N.x, N.y, a_y * N.z));
//		return N;
//	}
//
//	__device__ float3 sampleBSDF(const float3& wo, float3& wi, float& pdf, CMJState& seed) {
//		float3 i = wo;
//		float3 n = { 0.0, 1.0, 0.0 };
//		/*
//		wi = hemisphere_sampling(rnd(seed), rnd(seed), pdf);
//		float3 o = wi;
//		float3 m = normalize(wi + wo);
//		*/
//		//Walter Sampling
//		//float3 m = walterSampling(rnd(seed), rnd(seed));
//
//		//Visible Normal Sampling
//		float2 xi = cmj_2d(seed);
//		float3 m = visibleNormalSampling(i,xi.x,xi.y);
//
//		float3 o = reflect(-wo, m);
//		wi = o;
//		if (wi.y < 0.0f) {
//			pdf = 1;
//			return { 0.0,0.0,0.0 };
//		}
//
//		float im = absdot(i, m);
//		float in = absdot(i, n);
//		float on = absdot(o, n);
//
//		float3 F = shlickFresnel(F0,wo,make_float3(0.0,1.0,0.0));
//		float G_ = shadowG(o, i);
//		float D_ = GGX_D(m);
//
//		float3 brdf = F * G_ * D_ / (4.0f * in * on);
//
//		if (isnan(brdf.x) || isnan(brdf.y) || isnan(brdf.z)) {
//			brdf = make_float3(0);
//			pdf = 1.0f;
//		}
//
//		//Walter sampling PDF
//		//pdf = D_ * BSDFMath::cosTheta(m) / (4.0f * absDot(m, o));
//
//		//Visible Normal Sampling PDF
//		pdf = D_ * shadowG_1(i) * im / (absdot(i, n) * 4.0f * absdot(m, o));
//
//		return brdf;
//	}
//
//	__device__ float3 evaluateBSDF(const float3& wo, const float3& wi) {
//		float3 i = wo;
//		float3 n = { 0.0, 1.0, 0.0 };
//		float3 m = normalize(wi + wo);
//		float3 o = wi;
//		if (wi.y < 0.0f) return { 0.0,0.0,0.0 };
//		if (wo.y < 0.0f) return { 0.0,0.0,0.0 };
//
//		float im = fmaxf(absdot(i, m), 0.0001);
//		float in = fmaxf(absdot(i, n), 0.0001);
//		float on = fmaxf(absdot(o, n), 0.0001);
//
//		float3 F = shlickFresnel(F0,wo,make_float3(0.0,1.0,0.0));
//		float G_ = shadowG(o, i);
//		float D_ = GGX_D(m);
//
//		float3 brdf = F * G_ * D_ / (4.0f * in * on);
//
//		if (isnan(brdf.x) || isnan(brdf.y) || isnan(brdf.z)) {
//
//			/*
//			printf("brdf (%f,%f,%f) \n", brdf.x, brdf.y, brdf.z);
//			printf("m (%f,%f,%f) \n", m.x, m.y, m.z);
//			printf("wo (%f,%f,%f) \n", wo.x, wo.y, wo.z);
//			printf("wi (%f,%f,%f) \n", wi.x, wi.y, wi.z);
//			printf("F (%f,%f,%f) \n", F.x, F.y, F.z);
//			printf("G_ (%f,%f,%f) \n", G_);
//			printf("D_ (%f,%f,%f) \n", D_);
//			printf("im %f \n",im);
//			printf("in %f \n",in);
//			printf("on %f \n",on);
//			*/
//			brdf = make_float3(0);
//
//		}
//
//
//		return brdf;
//	}
//
//	__device__ float pdfBSDF(const float3& wo, const float3& wi) {
//		float3 i = wo;
//		float3 m = normalize(wi + wo);
//		float3 o = wi;
//		float3 n = make_float3(0, 1, 0);
//		float im = absdot(i, m);
//		float D_ = GGX_D(m);
//		return D_ * shadowG_1(i) * im / (absdot(i, n) * 4.0f * absdot(m, o));
//	}
//
//	//Woが決定した時点でウェイトとしてかかる値
//	__device__ float reflect_weight(const float3& wo) {
//		return  0.5f;
//	}
//};

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

	//Lambert lam;
	//GGX ggx;
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
		//lam = Lambert(basecolor);
		//ggx = GGX(basecolor, 0.1f);
		ior = 1.5;
		idealglass = MetaMaterialGlass(make_float3(1.0), ior);
		disney = DisneyBRDF(pyload);
	}

	__device__ float3 evaluateBSDF(const float3& wo,const float3& wi) {
		return disney.evaluateBSDF(wo, wi);
	}

	__device__ float3 sampleBSDF(const float3& wo, float3& wi, float& pdf, CMJState& state) {
		return disney.sampleBSDF(wo, wi, pdf, state);
	}

	__device__ float getPDF(const float3& wo, const float3& wi) {
		return disney.getPDF(wo, wi);
	}
};
