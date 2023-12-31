#pragma once

#include <cuda/helpers.h>
#include <sutil/vec_math.h>

#include <common/constant.h>
#include <kernel/cmj.h>
#include <kernel/math.h>
#include <kernel/Payload.h>

static __forceinline__ __device__ float3 getLUT(float2 uv) {
	return make_float3(tex2D<float4>(params.lut_texture,uv.x,uv.y));
	
}

class DisneyBRDF {
private:
	float3 m_basecolor;
	float m_alpha;
	float m_anisotropic;
	float m_subsurface;
	float m_metallic;
	float m_sheen;
	float m_clearcoat;
	float m_clearcoatGloss;
	float m_clearcoatAlpha;
	bool m_is_thinfilm;

	//Cosine Importance Sampling
	__forceinline__ __device__ float3 sampleDiffuse(const float2& uv, float& pdf) {
		float theta = 0.5f * acos(1.0f - 2.0f * uv.x);
		float phi = 2.0f * M_PIf * uv.y;
		float cosTheta = cos(theta);
		float sinTheta = sin(theta);
		float3 wi = make_float3(cos(phi) * sinTheta, cosTheta, sin(phi) * sinTheta);
		pdf = getPDFDiffuse(wi);
		return wi;
	}

	__forceinline__ __device__ float getPDFDiffuse(const float3& wi) {
		return fabsf(wi.y) * INV_PI;
	}

	__forceinline__ __device__ float GGX_D(const float3& wm) {
		float term1 = wm.x * wm.x / (m_alpha * m_alpha) + wm.z * wm.z / (m_alpha * m_alpha) + wm.y * wm.y;
		float term2 = PI * m_alpha * m_alpha * term1 * term1;
		return 1.0f / term2;
	}

	__forceinline__ __device__ float GGX_G1(const float3& w) {
		return 1.0f / (1.0f + GGX_Lambda(w));
	}

	__forceinline__ __device__ float GGX_G2_HeightCorrelated(const float3& wi, const float3& wo) {
		return 1.0f / (1.0f + GGX_Lambda(wi) + GGX_Lambda(wo));
	}

	__forceinline__ __device__ float GGX_Lambda(const float3& w) {
		float delta = 1.0f + (m_alpha * m_alpha * w.x * w.x + m_alpha * m_alpha * w.z * w.z) / (w.y * w.y);
		return (-1.0f + sqrtf(delta)) * 0.5f;
	}

	//https://arxiv.org/pdf/2306.05044.pdf
	__forceinline__ __device__ float3 sampleVisibleNormal(const float2& uv, const float3& wo) {
		float3 strech_wo = normalize(make_float3(wo.x * m_alpha, wo.y, wo.z * m_alpha));

		float phi = 2.0f * PI * uv.x;
		float z = fma((1.0f - uv.y), (1.0f + strech_wo.y), -strech_wo.y);
		float sinTheta = sqrtf(clamp(1.0f - z * z, 0.0f, 1.0f));
		float x = sinTheta * cos(phi);
		float y = sinTheta * sin(phi);

		float3 c = make_float3(x, z, y);

		float3 h = c + strech_wo;

		float3 wm = normalize(make_float3(h.x * m_alpha, h.y, h.z * m_alpha));

		return wm;
	}

	__forceinline__ __device__ float3 sampleSpecular(const float2& uv, const float3& wo, float& pdf) {
		float3 wm = sampleVisibleNormal(uv, wo);
		pdf = getPDFSpecular(wm, wo);
		return wm;
	}

	__forceinline__ __device__ float getPDFSpecular(const float3& wm, const float3& wo) {
		return 0.25f * GGX_D(wm) * GGX_G1(wo) * absdot(wo, wm) / (absdot(wm, wo) * fabsf(wo.y));
	}


	__forceinline__ __device__ float3 sampleClearcoat(const float2& uv, const float3& wo, float& pdf) {
		float cosineTheta = sqrtf(fmaxf((1.0f - powf(m_clearcoatAlpha * m_clearcoatAlpha, 1.0f - uv.x)) / (1.0f - m_clearcoatAlpha * m_clearcoatAlpha), 0.0f));
		float sinTheta = sqrtf(fmaxf(1.0f - cosineTheta * cosineTheta, 0.0f));
		float phi = PI2 * uv.y;
		float3 wm = make_float3(cos(phi) * sinTheta, cosineTheta, sin(phi) * sinTheta);
		pdf = getPDFClearcoat(wm, wo);
		return wm;
	}

	__forceinline__ __device__ float getPDFClearcoat(const float3& wm, const float3& wo) {
		return clearcoat_D(wm, m_clearcoatAlpha) * fabsf(wm.y) / (4.0f * fabsf(dot(wm, wo)));
	}

	__forceinline__ __device__ float f_tSchlick(float wn, float F90) {
		float delta = fmaxf(1.0f - wn, 0.0f);
		return 1.0f + (F90 - 1.0f) * delta * delta * delta * delta * delta;
	}

	//Specular
	__forceinline__ __device__ float3 specular(const float3& wo, const float3& wi, const float3& F0) {
		float3 wm = normalize(wo + wi);

		float ggxD = GGX_D(wm);
		float ggxG = GGX_G2_HeightCorrelated(wi, wo);
		float3 ggxF = shlickFresnel(F0, wo, wm);

		return 0.25f * ggxF * ggxD * ggxG / (fabsf(wo.y) * fabsf(wi.y));
	}

	__forceinline__ __device__ float clearcoat_G2_HeightCorrelated(const float3& wi, const float3& wo, const float& alpha) {
		return 1.0f / (1.0f + clearcoat_Lambda(wi, alpha) + clearcoat_Lambda(wo, alpha));
	}

	__forceinline__ __device__ float clearcoat_Lambda(const float3& w, const float& alpha) {
		float term1 = 1.0f + (alpha * alpha * w.x * w.x + alpha * alpha * w.z * w.z) / (w.y * w.y);
		return 0.5f * (-1.0f + sqrtf(term1));
	}

	__forceinline__ __device__ float clearcoat_D(const float3& wm, const float& alpha) {
		//const float cosine_wm = fabsf(wm.y);
		//const float term1 = PI * logf(alpha * alpha) * (1.0f + (cosine_wm * cosine_wm) * (alpha * alpha - 1.0f));
		//return (alpha * alpha - 1.0) / term1;

		float alpha2 = alpha * alpha;
		float t = 1.0f + (alpha2 - 1.0f) * wm.y * wm.y;
		return (alpha2 - 1.0f) / (PI * logf(alpha2) * t);
	}

	//Clearcoat
	__forceinline__ __device__ float3 clearcoat(const float3& wo, const float3& wi, const float clearcoat_alpha) {
		const float3 wm = normalize(wo + wi);

		float clearcoatD = clearcoat_D(wm, clearcoat_alpha);
		float clearcoatG = clearcoat_G2_HeightCorrelated(wi, wo, 0.25f);
		float3 clearcoatF = shlickFresnel(make_float3(0.04f), wo, wm);

		return 0.25f * clearcoatD * clearcoatG * clearcoatF / (fabsf(wo.y) * fabsf(wi.y));
	}


public:
	__device__ DisneyBRDF() {
		m_basecolor = make_float3(1);
		m_alpha = 0.5f;
		m_anisotropic = 0.0f;
		m_metallic = 0.0f;
		m_sheen = 0.0f;
		m_clearcoat = 0.0f;
		m_clearcoatGloss = 1.0f;
		m_clearcoatAlpha = 1.0f;
	}

	__device__ DisneyBRDF(const Payload& prd) {
		m_basecolor = prd.basecolor;
		//m_basecolor = make_float3(1.0,1.0,1.0);
		m_alpha = clamp(prd.roughness * prd.roughness, 0.01f, 1.0f);
		m_anisotropic = 0.0f;
		m_subsurface = 0.0f;
		m_metallic = prd.metallic;
		m_sheen = prd.sheen;
		m_clearcoat = prd.clearcoat;
		m_clearcoatGloss = 1.0f;
		m_clearcoatAlpha = lerp(0.1f, 0.001f, m_clearcoatGloss);
		m_is_thinfilm = prd.is_thinfilm;
	}

	__device__ float3 evaluateBSDF(const float3& wo, const float3& wi) {
		float3 f_diffuse;
		float3 f_subsurface;
		float3 f_sheen;
		float3 f_specular;
		float3 f_clearcoat;

		float3 wm = normalize(wo + wi);

		float dot_wi_n = fabsf(wi.y);
		float dot_wo_n = fabsf(wi.y);

		float cosine_d = absdot(wi, wm);
		float F_D90 = 0.5f + 2.0f * m_alpha * cosine_d * cosine_d;
		float F_SS90 = m_alpha * cosine_d * cosine_d;

		float f_tsi = f_tSchlick(dot_wi_n, F_D90);
		float f_tso = f_tSchlick(dot_wo_n, F_D90);

		//Diffuse		
		{
			f_diffuse = m_basecolor * f_tsi * f_tso * INV_PI;
		}

		//Subsurface
		{
			float deltacos = 1.0f / (dot_wi_n + dot_wo_n) - 0.5f;
			f_subsurface = m_basecolor * INV_PI * 1.25f * (f_tsi * f_tso * deltacos + 0.5f);
		}

		//Specular
		{
			float3 F0 = lerp(make_float3(0.08f), m_basecolor, m_metallic);

			if (m_is_thinfilm) {
				float thickness = m_basecolor.x;
				float cosine = absdot(wi, wm);
				F0 = getLUT(make_float2(thickness,cosine));
			}
			f_specular = specular(wo, wi, F0);
		}

		//Sheen
		{
			float delta = fmaxf(1.0f - absdot(wi, wm), 0.0f);
			f_sheen = m_sheen * make_float3(1.0f) * delta * delta * delta * delta * delta;
		}

		//Clearcoat
		{
			f_clearcoat = 0.25f * clearcoat(wo, wi, m_clearcoatAlpha);
		}

		//return make_float3(m_clearcoat);
		return (lerp(f_diffuse, f_subsurface, m_subsurface) + f_sheen) * (1.0f - m_metallic) + f_specular + f_clearcoat * m_clearcoat;
		//return f_clearcoat * m_clearcoat;
	}

	__device__ float3 sampleBSDF(const float3& wo, float3& wi, float& pdf, CMJState& state) {
		//TODO nantokashitai
		float diffuseWeight = 1.0f * (1.0f - m_metallic);
		float specularWeight = 0.5f;
		float clearcoatWeight = 0.0f;

		float sumWeight = diffuseWeight + specularWeight + clearcoatWeight;
		//float sumWeight = diffuseWeight + specularWeight;
		float dw = diffuseWeight / sumWeight;
		float sw = specularWeight / sumWeight;
		float cw = clearcoatWeight / sumWeight;

		float select_p = cmj_1d(state);

		float pdf_diffuse = 1.0f;
		float pdf_specular = 1.0f;
		float pdf_clearcoat = 1.0f;
		float2 xi = cmj_2d(state);

		//if (select_p < dw) {
		//	//Diffuse
		//	wi = sampleDiffuse(xi, pdf_diffuse);
		//	float3 wm = normalize(wi + wo);

		//	pdf_specular = getPDFSpecular(wm, wo);
		//}
		//else {
		//	//Specular
		//	float3 wm = sampleSpecular(xi, wo, pdf_specular);
		//	wi = reflect(-wo, wm);

		//	pdf_diffuse = getPDFDiffuse(wi);
		//}

		//pdf = dw * pdf_diffuse + sw * pdf_specular;

		if (select_p < dw) {
			//Diffuse
			wi = sampleDiffuse(xi, pdf_diffuse);
			float3 wm = normalize(wi + wo);
			pdf_specular = getPDFSpecular(wm, wo);
			pdf_clearcoat = getPDFClearcoat(wm, wo);

		}
		else if (select_p < dw + sw) {
			//Specular
			float3 wm = sampleSpecular(xi, wo, pdf_specular);
			wi = reflect(-wo, wm);

			pdf_diffuse = getPDFDiffuse(wi);
			pdf_clearcoat = getPDFClearcoat(wm, wo);

		}
		else {
			//Clearcoat
			float3 wm = sampleClearcoat(xi, wo, pdf_clearcoat);
			wi = reflect(-wo, wm);

			pdf_diffuse = getPDFDiffuse(wi);
			pdf_specular = getPDFSpecular(wm, wo);
		}

		pdf = dw * pdf_diffuse + sw * pdf_specular + cw * pdf_clearcoat;

		if (wi.y < 0.0) {
			pdf = 1.0f;
			return make_float3(0.0);
		}

		return evaluateBSDF(wo, wi);
	}

	__device__ float getPDF(const float3& wo, const float3& wi) {
		//Clearcoat多分使わないので放置
		float diffuseWeight = 1.0f * (1.0f - m_metallic);
		float specularWeight = 0.5f;
		float clearcoatWeight = 0.0f;

		float sumWeight = diffuseWeight + specularWeight + clearcoatWeight;
		float dw = diffuseWeight / sumWeight;
		float sw = specularWeight / sumWeight;
		float cw = clearcoatWeight / sumWeight;

		float3 wm = normalize(wo + wi);
		float pdf_diffuse = getPDFDiffuse(wi);
		float pdf_specular = getPDFSpecular(wm, wo);
		//float pdf_clearcoat = getPDFClearcoat(wm, wo);

		return dw * pdf_diffuse + sw * pdf_specular;
	}
};
