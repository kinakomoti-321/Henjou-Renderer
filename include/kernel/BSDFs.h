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

	__forceinline__ __device__ MetaMaterialGlass(const float3& rho, const float& ior) :rho_(rho), ior_(ior) {

	}

	__forceinline__ __device__ float3 sampleBSDF(const float3& wo, float3& wi, float& pdf, CMJState& state) {
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

	__forceinline__ __device__ float3 evalueateBSDF(const float3& wo, const float3& wi) {

		return make_float3(0);
	}

	__forceinline__ __device__ float pdfBSDF(const float3& wo, const float3& wi) {
		return 0;
	}
};


//Heitz 2017 
class EnagyConservationGGX {
	float3 F0;
	float alpha;

	__device__ float P1(const float h)const {
		//Uniform
		const float value = (h >= -1.0f && h <= 1.0f) ? 0.5f : 0.0f;
		return value;
	};

	__device__ float C1(const float h)const {
		
		//Uniform
		const float value = fmin(1.0f, fmax(0.0f, 0.5f * (h + 1.0f)));
		return value;

	};

	__device__ float invC1(const float U)const {
		const float h =fmax(-1.0f, fmin(1.0f, 2.0f * U - 1.0f));
		return h;
	}

	//__device__ float GGX_D(const float3& wm) const {
	//	float term1 = wm.x * wm.x / (alpha * alpha) + wm.z * wm.z / (alpha * alpha) + wm.y * wm.y;
	//	float term2 = PI * alpha * alpha * term1 * term1;
	//	return 1.0f / term2;
	//}

	//__device__ float GGX_G1(const float3& w) const {
	//	return 1.0f / (1.0f + GGX_Lambda(w));
	//}

	//__device__ float GGX_G2_HeightCorrelated(const float3& wi, const float3& wo) const {
	//	return 1.0f / (1.0f + GGX_Lambda(wi) + GGX_Lambda(wo));
	//}

	//__device__ float GGX_Lambda(const float3& v) const {
	//	float delta = 1.0f + (alpha * alpha * v.x * v.x + alpha * alpha * v.z * v.z) / (v.y * v.y);
	//	return (-1.0 + sqrtf(delta)) / 2.0f;
	//	//float term1 = (w.x * w.x / (alpha * alpha) + w.z * w.z / (alpha * alpha)) / (w.y * w.y);
	//	//return 0.5f * (-1.0f + sqrtf(term1));
	//}
	__device__ float sign(float x) const{
		return ((x > 0.0f) ? 1.0f : -1.0f);
	}

	__device__ float GGX_Lambda(const float3& wi) const
	{
		if (wi.y > 0.9999f)
			return 0.0f;
		if (wi.y < -0.9999f)
			return -1.0f;

		// a
		const float theta_i = acosf(wi.z);
		const float a = 1.0f / (tanf(theta_i) * alpha);

		// value
		const float value = 0.5f * (-1.0f + sign(a) * sqrtf(1 + 1 / (a * a)));
		return value;
	}

	__device__ float G_1_Height(const float3& wi,const float h0) const {
		if (wi.y > 0.9999f)
			return 1.0f;
		if (wi.y <= 0.0f)
			return 0.0f;
		// height CDF
		const float C1_h0 = C1(h0);
		// Lambda
		const float Lambda = GGX_Lambda(wi);
		// value
		const float value = powf(C1_h0, Lambda);
		return value;
	}


	__device__ float sampleHeight(const float3& wr, const float hr, const float U) const {

		if (wr.y > 0.9999f)
			return FLT_MAX;
		if (wr.y < -0.9999f)
		{
			const float value = invC1(U * C1(hr));
			return value;
		}
		if (fabsf(wr.y) < 0.0001f)
			return hr;

		// probability of intersection
		const float G_1_ = G_1_Height(wr, hr);
		if (U > 1.0f - G_1_) // leave the microsurface
			return FLT_MAX;
		const float h = invC1(
			C1(hr) / powf((1.0f - U), 1.0f / GGX_Lambda(wr))
		);
		return h;
	}

	__device__ float3 visibleNormalSampling(const float3& V_, float u, float v) {
		float a_x = alpha, a_y = alpha;
		float3 V = normalize(make_float3(a_x * V_.x, V_.y, a_y * V_.z));

		float3 n = make_float3(0, 1, 0);
		if (V.y > 0.99) n = make_float3(1, 0, 0);
		float3 T1 = normalize(cross(V, n));
		float3 T2 = normalize(cross(T1, V));

		float r = sqrtf(u);
		float a = 1.0f / (1.0f + V.y);
		float phi;
		if (a > v) {
			phi = PI * v / a;
		}
		else {
			phi = PI * (v - a) / (1.0f - a) + PI;
		}

		float P1 = r * cosf(phi);
		float P2 = r * sinf(phi);
		if (a < v) P2 *= V.y;

		float3 N = P1 * T1 + P2 * T2 + sqrtf(fmaxf(1.0f - P1 * P1 - P2 * P2, 0.0f)) * V;

		N = normalize(make_float3(a_x * N.x, N.y, a_y * N.z));
		return N;
	}
//__device__ float3 sampleVisibleNormal(float2 uv, float3 wo) {
//	float3 strech_wo = normalize(make_float3(wo.x * alpha, wo.y, wo.z * alpha));
//
//	float phi = 2.0f * PI * uv.x;
//	float z = fma((1.0f - uv.y), (1.0f + strech_wo.y), -strech_wo.y);
//	float sinTheta = sqrtf(clamp(1.0f - z * z, 0.0f, 1.0f));
//	float x = sinTheta * cos(phi);
//		float y = sinTheta * sin(phi);
//
//		float3 c = make_float3(x, z, y);
//
//		float3 h = c + strech_wo;
//
//		float3 wm = normalize(make_float3(h.x * alpha, h.y, h.z * alpha));
//
//		return wm;
//	}

	//__device__ float evalPhaseFunction(const float3& wi, const float3& wo) {
	//	const float3 wh = normalize(wi + wo);
	//	if (wh.z < 0.0f) return 0.0;

	//	float D = GGX_D(wh);
	//	float G_1 = GGX_G1(wi);

	//	const float value = 0.25f *  D * G_1 * fabsf(dot(wi, wh)) / (wi.y * fabsf(dot(wi,wh)));

	//	return value;
	//}
	//__device__ static bool IsFiniteNumber(float x)
	//{
	//	return (x <= FLT_MAX && x >= -FLT_MAX);
	//}
	//
	//float2 sampleP22_11(const float theta_i, const float U, const float U_2) const
	//{
	//	float2 slope;

	//	if (theta_i < 0.0001f)
	//	{
	//		const float r = sqrtf(U / (1.0f - U));
	//		const float phi = 6.28318530718f * U_2;
	//		slope.x = r * cosf(phi);
	//		slope.y = r * sinf(phi);
	//		return slope;
	//	}

	//	// constant
	//	const float sin_theta_i = sinf(theta_i);
	//	const float cos_theta_i = cosf(theta_i);
	//	const float tan_theta_i = sin_theta_i / cos_theta_i;

	//	// slope associated to theta_i
	//	const float slope_i = cos_theta_i / sin_theta_i;

	//	// projected area
	//	const float projectedarea = 0.5f * (cos_theta_i + 1.0f);
	//	if (projectedarea < 0.0001f || projectedarea != projectedarea)
	//		return make_float2(0, 0);
	//	// normalization coefficient
	//	const float c = 1.0f / projectedarea;

	//	const float A = 2.0f * U / cos_theta_i / c - 1.0f;
	//	const float B = tan_theta_i;
	//	const float tmp = 1.0f / (A * A - 1.0f);

	//	const float D = sqrtf(std::max(0.0f, B * B * tmp * tmp - (A * A - B * B) * tmp));
	//	const float slope_x_1 = B * tmp - D;
	//	const float slope_x_2 = B * tmp + D;
	//	slope.x = (A < 0.0f || slope_x_2 > 1.0f / tan_theta_i) ? slope_x_1 : slope_x_2;

	//	float U2;
	//	float S;
	//	if (U_2 > 0.5f)
	//	{
	//		S = 1.0f;
	//		U2 = 2.0f * (U_2 - 0.5f);
	//	}
	//	else
	//	{
	//		S = -1.0f;
	//		U2 = 2.0f * (0.5f - U_2);
	//	}
	//	const float z = (U2 * (U2 * (U2 * 0.27385f - 0.73369f) + 0.46341f)) / (U2 * (U2 * (U2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
	//	slope.y = S * z * sqrtf(1.0f + slope.x * slope.x);

	//	return slope;
	//}

	//float3 sampleD_wi(const float3& wi, const float U1, const float U2) const {
	//	float m_alpha_x = alpha;
	//	float m_alpha_y = alpha;
	//	// stretch to match configuration with alpha=1.0	
	//	const float3 wi_11 = normalize(make_float3(m_alpha_x * wi.x, m_alpha_y * wi.y, wi.z));

	//	// sample visible slope with alpha=1.0
	//	float2 slope_11 = sampleP22_11(acosf(wi_11.z), U1, U2);

	//	// align with view direction
	//	const float phi = atan2(wi_11.y, wi_11.x);
	//	float2 slope = make_float2(cosf(phi) * slope_11.x - sinf(phi) * slope_11.y, sinf(phi) * slope_11.x + cos(phi) * slope_11.y);

	//	// stretch back
	//	slope.x *= m_alpha_x;
	//	slope.y *= m_alpha_y;

	//	// if numerical instability
	//	if ((slope.x != slope.x) || !IsFiniteNumber(slope.x))
	//	{
	//		if (wi.z > 0) return make_float3(0.0f, 0.0f, 1.0f);
	//		else return normalize(make_float3(wi.x, wi.y, 0.0f));
	//	}

	//	// compute normal
	//	const float3 wm = normalize(make_float3(-slope.x, -slope.y, 1.0f));
	//	return wm;
	//}

	__device__ float3 samplePhaseFunction(const float3& wi,CMJState& state){
		const float2 uv = cmj_2d(state);
		float3 wm = visibleNormalSampling(wi,uv.x,uv.y);
		const float3 wo = -wi + 2.0 * wm * dot(wi,wm);
		return wo;
	}

	//__device__ float eval(const float3& wi, const float3& wo,CMJState& state ,const int scatteringOrder){
	//	if (wo.z < 0) return 0;
	//	float3 wr = -wi;
	//	float hr = 1.0f + invC1(0.999f);
	//	
	//	float sum = 0.0f;
	//	int current_scatteringOrder = 0;
	//	while (scatteringOrder == 0 || current_scatteringOrder <= current_scatteringOrder) {
	//		float U = cmj_1d(state);
	//		hr = sampleHeight(wr, hr, U);

	//		if (hr == FLT_MAX)
	//			break;
	//		else
	//			current_scatteringOrder++;	
	//		
	//		float phasefunction = evalPhaseFunction(wi,wo);
	//		float shadowing = G_1_Height(wo, hr);
	//		float I = phasefunction * shadowing;

	//		if (isinf(I) && (scatteringOrder == 0 || current_scatteringOrder == current_scatteringOrder))
	//		{
	//			sum += I;
	//		}

	//		wr = samplePhaseFunction(-wr,state);

	//		if (isnan(hr) || isnan(wr.z)) {
	//			return 0.0f;
	//		}
	//	}

	//	return sum;
	//}
	
	//Importance Sampling
	__device__ float3 sample(const float3& wi,float3& wo, int& scatteringOrder,CMJState& state) 
	{
		// init
		float3 wr = -wi;
		float hr = 1.0f + invC1(0.999f);
		// random walk
		scatteringOrder = 0;
		float3 weight = make_float3(1.0);
		while (true)
		{
			// next height
			float U = cmj_1d(state);
			hr = sampleHeight(wr, hr, U);
			// leave the microsurface?
			if (hr == FLT_MAX) {
				//printf("test");
				break;
			}
			else
				scatteringOrder++;
			// next direction
			float3 nextwr = samplePhaseFunction(-wr,state);
			float3 hv = normalize( -wr + nextwr);
			float3 Fres = shlickFresnel(F0, nextwr, hv);

			weight *= Fres;
			wr = nextwr;

			// if NaN (should not happen, just in case)
			if ((hr != hr) || (wr.z != wr.z))
				return make_float3(0, 0, 1);
		}
		wo = wr;
		return weight;
	}

	public:
	__device__ EnagyConservationGGX() {
		F0 = { 0.04,0.04,0.04 };
		alpha = 0.5;
	}

	__device__ EnagyConservationGGX(float3 iF0, float iroughness) {
		F0 = iF0;
		alpha = clamp(iroughness * iroughness, 0.001f, 1.0f);
	}

	//__device__ float3 evaluateBSDF(const float3& wo, const float3& wi) {
	//	const float3 wm = normalize(wo + wi);

	//	float ggxD = GGX_D(wm);
	//	float ggxG2 = GGX_G2_HeightCorrelated(wi, wo);
	//	float3 ggxF = shlickFresnel(F0, wi, wm);

	//	return ggxD * ggxG2 * ggxF / (4.0 * wo.y * wi.y);
	//}

	__device__ float3 sampleBSDF(const float3& wo, float3& wi, CMJState& state,float& pdf) {
		int scatteringOrder;
		float3 bsdf = sample(wo,wi,scatteringOrder,state);
		if (wi.y < 0.0) {
			return make_float3(0.0);
		}
		pdf = fabsf(wi.y);
		return bsdf;
	}
};

class FastMultipleGGX {

};

class BSDF {
private:
	bool is_specular = false;
	bool is_ggx = false;
	//Lambert lam;
	//GGX ggx;
	MetaMaterialGlass idealglass;
	EnagyConservationGGX eggx;
	DisneyBRDF disney;

public:
	__device__ BSDF() {
		is_specular = false;
	}

	__device__ BSDF(const Payload& pyload) {
		is_specular = pyload.is_specular;
		float ior = pyload.ior;
		idealglass = MetaMaterialGlass(make_float3(1.0), ior);
		disney = DisneyBRDF(pyload);
		eggx = EnagyConservationGGX(pyload.basecolor, pyload.roughness);
		//ggx = GGX(pyload.basecolor, pyload.roughness);
		is_ggx = pyload.metallic > 0.5;
	}

	__device__ float3 evaluateBSDF(const float3& wo,const float3& wi) {
		if (is_specular) {
			return idealglass.evalueateBSDF(wo, wi);
		}
		else {
			return disney.evaluateBSDF(wo, wi);
		}
	}

	__device__ float3 sampleBSDF(const float3& wo, float3& wi, float& pdf, CMJState& state) {
		if (is_specular) {
			return idealglass.sampleBSDF(wo, wi, pdf, state);
		}
		else {
				//return disney.sampleBSDF(wo, wi, pdf, state);
			if (!is_ggx) {
				return disney.sampleBSDF(wo, wi, pdf, state);
			}
			else {
				return eggx.sampleBSDF(wo, wi, state, pdf);
				//return ggx.sampleBSDF(wo, wi, pdf,state);
			}
		}
	}

	__device__ float getPDF(const float3& wo, const float3& wi) {
		if (is_specular) {
			return idealglass.pdfBSDF(wo, wi);
		}
		return disney.getPDF(wo, wi);
	}
};
