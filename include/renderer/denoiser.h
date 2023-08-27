#pragma once
#pragma once
#include <optix.h>
#include <sutil/sutil.h>
#include <sutil/Exception.h>

static OptixImage2D createOptixImage2D(unsigned int width, unsigned int height, float4* data)
{
	OptixImage2D oi;

	oi.width = width;
	oi.height = height;
	oi.rowStrideInBytes = width * sizeof(float4);
	oi.pixelStrideInBytes = sizeof(float4);
	oi.data = reinterpret_cast<CUdeviceptr>(data);
	oi.format = OPTIX_PIXEL_FORMAT_FLOAT4;

	return oi;
}

static OptixImage2D createOptixImage2D(unsigned int width, unsigned int height, float2* data)
{
	OptixImage2D oi;

	oi.width = width;
	oi.height = height;
	oi.rowStrideInBytes = width * sizeof(float2);
	oi.pixelStrideInBytes = sizeof(float2);
	oi.data = reinterpret_cast<CUdeviceptr>(data);
	oi.format = OPTIX_PIXEL_FORMAT_FLOAT2;

	return oi;
}

enum DenoiseType {
	NONE,
	TEMPORAL,
	UPSCALE2X,
};

class OptixDenoiserManager {
private:
	OptixDeviceContext context = nullptr;
	OptixDenoiser denoiser = nullptr;
	CUstream cu_stream = nullptr;

	CUdeviceptr m_scratch = 0;
	uint32_t m_scratch_size = 0;
	CUdeviceptr m_state = 0;
	uint32_t m_state_size = 0;

	unsigned int in_width_ = 0;
	unsigned int in_height_ = 0;
	unsigned int out_width_ = 0; 
	unsigned int out_height_ = 0;

	float4* albedo;
	float4* normal;
	float4* input;
	float4* output;

	DenoiseType denoise_type = NONE;

public:

	OptixDenoiserManager(const unsigned int& in_width, const unsigned int& in_height, const unsigned int& out_width,const unsigned int& out_height,
		OptixDeviceContext context, CUstream& cu_stream, DenoiseType denoise_type) : context(context), cu_stream(cu_stream), denoise_type(denoise_type) {

		OptixDenoiserOptions options;
		options.guideAlbedo = 1;
		options.guideNormal = 1;

		OptixDenoiserModelKind model_kind;
		in_width_ = in_width;
		in_height_ = in_height;

		out_width_ = in_width_;
		out_height_ = in_height_;

		switch (denoise_type)
		{
		case NONE:
			model_kind = OPTIX_DENOISER_MODEL_KIND_HDR;
			break;
		case TEMPORAL:
			model_kind = OPTIX_DENOISER_MODEL_KIND_TEMPORAL;
			break;
		case UPSCALE2X:
			model_kind = OPTIX_DENOISER_MODEL_KIND_UPSCALE2X;
			break;
		default:
			model_kind = OPTIX_DENOISER_MODEL_KIND_LDR;
			model_kind = OPTIX_DENOISER_MODEL_KIND_TEMPORAL;
			break;
		}


		OPTIX_CHECK(optixDenoiserCreate(
			context,
			model_kind,
			&options,
			&denoiser
		));

		//Setup
		{
			OptixDenoiserSizes denoiser_size;
			OPTIX_CHECK(optixDenoiserComputeMemoryResources(
				denoiser,
				out_width_,
				out_height_,
				&denoiser_size
			));

			m_scratch_size = static_cast<uint32_t>(denoiser_size.withOverlapScratchSizeInBytes);
			m_state_size = static_cast<uint32_t>(denoiser_size.stateSizeInBytes);

			CUDA_CHECK(cudaMalloc(
				reinterpret_cast<void**>(&m_scratch),
				m_scratch_size
			));

			CUDA_CHECK(cudaMalloc(
				reinterpret_cast<void**>(&m_state),
				m_state_size
			));

			OPTIX_CHECK(optixDenoiserSetup(
				denoiser,
				cu_stream,
				in_width,
				in_height,
				m_state,
				m_state_size,
				m_scratch,
				m_scratch_size
			));
		}
	}

	void layerSet(float4* in_albedo, float4* in_normal, float4* in_input, float4* in_output) {
		albedo = in_albedo;
		normal = in_normal;
		input = in_input;
		output = in_output;
	}

	void denoise() {
		OptixDenoiserGuideLayer guidelayer;
		guidelayer.albedo = createOptixImage2D(in_width_, in_height_, albedo);
		guidelayer.normal = createOptixImage2D(in_width_, in_height_, normal);

		OptixDenoiserLayer layers;
		layers.input = createOptixImage2D(in_width_, in_height_, input);
		layers.output = createOptixImage2D(out_width_, out_height_, output);

		OptixDenoiserParams param;
		param.blendFactor = 0.0f;
		param.hdrIntensity = reinterpret_cast<CUdeviceptr>(nullptr);
		param.hdrAverageColor = reinterpret_cast<CUdeviceptr>(nullptr);

		OPTIX_CHECK(optixDenoiserInvoke(
			denoiser,
			cu_stream,
			&param,
			m_state,
			m_state_size,
			&guidelayer,
			&layers,
			1,
			0,
			0,
			m_scratch,
			m_scratch_size
		));
	}

	~OptixDenoiserManager() {
		OPTIX_CHECK(optixDenoiserDestroy(denoiser));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_scratch)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state)));
	}
};
