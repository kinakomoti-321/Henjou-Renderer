#pragma once

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>

#include <array>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <sutil/Camera.h>
#include <sutil/Trackball.h>

#include <external/glm/glm/glm.hpp>

#include <HenjouRenderer/henjouRenderer.h>
#include <renderer/denoiser.h>
#include <loader/gltfloader.h>
#include <file_reader.h>
#include <renderer/scene.h>
#include <renderer/material.h>
#include <loader/objloader.h>
#include <cu/cuda_buffer.h>
#include <cu/matrix_4x3.h>
#include <common/log.h>
#include <common/timer.h>
#include <renderer/render_option.h>
#include <loader/render_json_loader.h>
#include <spdlog/spdlog.h>

template <typename T>
struct SbtRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;

void configureCamera(sutil::Camera& cam, const uint32_t width, const uint32_t height)
{
	cam.setEye({ 1.0f, 0.0f, 0.0f });
	cam.setLookat({ 0.0f, 0.0f, 0.0f });
	cam.setUp({ 0.0f, 1.0f, 0.0 });
	cam.setFovY(45.0f);
	cam.setAspectRatio((float)width / (float)height);
}

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
	std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
		<< message << "\n";
}

struct GASData {
	OptixTraversableHandle handle;
	CUdeviceptr            d_gas_output_buffer;
};

inline float3 toSRGB(const float3& col) {
	float  invGamma = 1.0f / 2.4f;
	float3 powed = make_float3(std::pow(col.x, invGamma), std::pow(col.y, invGamma), std::pow(col.z, invGamma));
	return make_float3(
		col.x < 0.0031308f ? 12.92f * col.x : 1.055f * powed.x - 0.055f,
		col.y < 0.0031308f ? 12.92f * col.y : 1.055f * powed.y - 0.055f,
		col.z < 0.0031308f ? 12.92f * col.z : 1.055f * powed.z - 0.055f);
}

inline unsigned char quantizeUnsignedChar(float x) {
	enum { N = (1 << 8) - 1, Np1 = (1 << 8) };
	return (unsigned char)std::min((unsigned int)(x * (float)Np1), (unsigned int)N);
}

void float4ConvertColor(float4* data, uchar4* color, unsigned int width, unsigned int height) {
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			unsigned int idx = i + width * j;
			float3 col = make_float3(data[idx]);
			col = toSRGB(col);

			color[idx] = make_uchar4(
				quantizeUnsignedChar(col.x),
				quantizeUnsignedChar(col.y),
				quantizeUnsignedChar(col.z),
				(unsigned char)255);
		}
	}
}

struct BufferObject {
	float4* buffer;
	unsigned int width;
	unsigned int height;
	CUdeviceptr d_gpu_buffer = 0;

	BufferObject(unsigned int in_width, unsigned int in_height) {
		width = in_width;
		height = in_height;

		buffer = new float4[in_width * in_height];

		const size_t buffer_size = sizeof(float4) * in_width * in_height;
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&d_gpu_buffer),
			buffer_size
		));
	}

	~BufferObject() {
		delete[] buffer;
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gpu_buffer)));
	}

	void cpyGPUBufferToHost() {
		const size_t buffer_size = sizeof(float4) * width * height;
		CUDA_CHECK(cudaMemcpy(
			buffer,
			reinterpret_cast<void*>(d_gpu_buffer),
			buffer_size,
			cudaMemcpyDeviceToHost
		));
	}
};

class Renderer {
private:
	RenderOption render_option_;
	SceneData scene_data_;

	//SceneData Device Buffer
	cuh::CUDevicePointer vertices_buffer_;
	cuh::CUDevicePointer indices_buffer_;
	cuh::CUDevicePointer texcoords_buffer_;
	cuh::CUDevicePointer normals_buffer_;
	cuh::CUDevicePointer material_ids_buffer_;
	cuh::CUDevicePointer colors_buffer_;
	cuh::CUDevicePointer prim_offset_buffer_;
	cuh::CUDevicePointer light_prim_ids_buffer_;
	cuh::CUDevicePointer light_prim_emission_buffer_;

	OptixDeviceContext optix_context_ = nullptr;

	OptixModule optix_module_ = nullptr;

	OptixPipeline optix_pipeline_ = nullptr;
	OptixPipelineCompileOptions pipeline_compile_options_ = {};

	OptixProgramGroup raygen_prog_group_ = nullptr;
	OptixProgramGroup miss_prog_group_ = nullptr;
	OptixProgramGroup hitgroup_prog_group_ = nullptr;
	OptixProgramGroup hitgroup_prog_shadow_group_ = nullptr;

	OptixShaderBindingTable optix_sbt_ = {};

	OptixTraversableHandle ias_handle_;
	CUdeviceptr d_ias_buffer_;

	std::vector<Matrix4x3> transform_matrices_;
	cuh::CUDevicePointer transform_matrices_buffer_;
	std::vector<Matrix4x3> inv_transform_matrices_;
	cuh::CUDevicePointer inv_transform_matrices_buffer_;

	std::vector<OptixTraversableHandle> gas_handle_;
	std::vector<CUdeviceptr> d_gas_buffer_;

	std::vector<cudaArray_t> carray_objects_;
	std::vector<cudaTextureObject_t> ctexture_objects_;
	cuh::CUDevicePointer d_texture_objects_;

	std::shared_ptr<HDRTexture> hdrtexture_ = nullptr;
	cudaArray_t ibl_texture_array_;
	cudaTextureObject_t ibl_texture_object_;
	
	//LUT
	std::shared_ptr<Texture> lut_texture_ = nullptr;
	cudaArray_t lut_texture_array_;
	cudaTextureObject_t lut_texture_object_;


	const unsigned int RAYTYPE_ = 2;

private:

	void cpySceneDataToDevice() {
		//Matrix initialize
		transform_matrices_.resize(scene_data_.instances.size());
		for (int i = 0; i < transform_matrices_.size(); i++) {
			Matrix4x3 mx;
			mx.r0 = { 1.0,0.0,0.0,0.0 };
			mx.r1 = { 0.0,1.0,0.0,0.0 };
			mx.r2 = { 0.0,0.0,1.0,0.0 };
			transform_matrices_[i] = mx;
		}

		inv_transform_matrices_.resize(scene_data_.instances.size());
		for (int i = 0; i < inv_transform_matrices_.size(); i++) {
			Matrix4x3 mx;
			mx.r0 = { 1.0,0.0,0.0,0.0 };
			mx.r1 = { 0.0,1.0,0.0,0.0 };
			mx.r2 = { 0.0,0.0,1.0,0.0 };
			inv_transform_matrices_[i] = mx;
		}

		//Copy to Device Memory
		vertices_buffer_.cpyHostToDevice(scene_data_.vertices);
		indices_buffer_.cpyHostToDevice(scene_data_.indices);
		normals_buffer_.cpyHostToDevice(scene_data_.normals);
		texcoords_buffer_.cpyHostToDevice(scene_data_.texcoords);
		material_ids_buffer_.cpyHostToDevice(scene_data_.material_ids);
		colors_buffer_.cpyHostToDevice(scene_data_.colors);
		prim_offset_buffer_.cpyHostToDevice(scene_data_.prim_offset);

		light_prim_ids_buffer_.cpyHostToDevice(scene_data_.light_prim_ids);
		light_prim_emission_buffer_.cpyHostToDevice(scene_data_.light_prim_emission);

		transform_matrices_buffer_.cpyHostToDevice(transform_matrices_);
		inv_transform_matrices_buffer_.cpyHostToDevice(inv_transform_matrices_);

		spdlog::info("Scene Data");
		spdlog::info("number of vertex : {:16d}", scene_data_.vertices.size());
		spdlog::info("number of index : {:16d}", scene_data_.indices.size());
		spdlog::info("number of normal : {:16d}", scene_data_.normals.size());
		spdlog::info("number of texcoord : {:16d}", scene_data_.texcoords.size());
		spdlog::info("number of material id : {:16d}", scene_data_.material_ids.size());
		spdlog::info("number of material : {:16d}", scene_data_.materials.size());
		spdlog::info("number of color : {:16d}", scene_data_.colors.size());
		spdlog::info("number of prim offset : {:16d}", scene_data_.prim_offset.size());
		spdlog::info("number of animation : {:16d}", scene_data_.animations.size());
		spdlog::info("number of instance : {:16d}", scene_data_.instances.size());
		spdlog::info("number of geometry : {:16d}", scene_data_.geometries.size());
		spdlog::info("number of light ID : {:16d}", scene_data_.light_prim_ids.size());
		spdlog::info("number of light Emission : {:16d}", scene_data_.light_prim_emission.size());

		//textures
		textureBind();

		//IBL
		setSky();

		//LUT;
		setLUT();
	}

	void updateIASMatrix(float time) {

		for (int i = 0; i < scene_data_.instances.size(); i++) {
			auto& inst = scene_data_.instances[i];
			unsigned int anim_id = inst.animation_id;
			auto& anim = scene_data_.animations[anim_id];
			Affine4x4 affine = anim.getAnimationAffine(time);

			Matrix4x3 mx;
			mx.r0 = { affine[0],affine[1],affine[2],affine[3] };
			mx.r1 = { affine[4],affine[5],affine[6],affine[7] };
			mx.r2 = { affine[8],affine[9],affine[10],affine[11] };

			transform_matrices_[i] = mx;

			glm::mat4x4 mat;
			mat[0] = { affine[0],affine[1],affine[2],affine[3] };
			mat[1] = { affine[4],affine[5],affine[6],affine[7] };
			mat[2] = { affine[8],affine[9],affine[10],affine[11] };
			mat[3] = { affine[12],affine[13],affine[14],affine[15] };

			glm::mat4x4 inv_mat = glm::inverse(mat);
			Matrix4x3 inv_mx;
			inv_mx.r0 = { inv_mat[0][0],inv_mat[0][1],inv_mat[0][2],inv_mat[0][3] };
			inv_mx.r1 = { inv_mat[1][0],inv_mat[1][1],inv_mat[1][2],inv_mat[1][3] };
			inv_mx.r2 = { inv_mat[2][0],inv_mat[2][1],inv_mat[2][2],inv_mat[2][3] };

			inv_transform_matrices_[i] = inv_mx;
		}

		buildIAS();

		transform_matrices_buffer_.updateCpyHostToDevice(transform_matrices_);
		inv_transform_matrices_buffer_.updateCpyHostToDevice(inv_transform_matrices_);
	}

	void optixDeviceContextInitialize() {
		CUDA_CHECK(cudaFree(0));

		OPTIX_CHECK(optixInit());

		OptixDeviceContextOptions options = {};
		options.logCallbackFunction = &context_log_cb;
		options.logCallbackLevel = 4;

		CUcontext cuCtx = 0;
		OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &optix_context_));



		size_t heap_size = 65536;
		CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, heap_size));
		CUDA_CHECK(cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize));
		spdlog::info("heap size : {:16d}", heap_size);
		spdlog::info("Optix Device Context Initialize");
	}

	void optixTraversalBuild() {
		buildGAS();
		buildIAS();
	}

	void buildGAS() {
		OptixAccelBuildOptions accel_options = {};
		accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
		accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

		gas_handle_.resize(scene_data_.geometries.size());
		d_gas_buffer_.resize(scene_data_.geometries.size());

		std::vector<uint32_t> triangle_input_flags(scene_data_.materials.size());
		for (int i = 0; i < scene_data_.materials.size(); i++) {
			triangle_input_flags[i] = OPTIX_GEOMETRY_FLAG_NONE;
		}

		spdlog::info("GAS Build");
		spdlog::info("GAS Number : {:5d}", scene_data_.geometries.size());

		Timer timer;
		timer.Start();
		for (int i = 0; i < scene_data_.geometries.size(); i++) {
			OptixBuildInput triangle_input = {};
			triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
			triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
			triangle_input.triangleArray.numVertices = scene_data_.vertices.size();
			triangle_input.triangleArray.vertexBuffers = vertices_buffer_.getDevicePtr();
			triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
			triangle_input.triangleArray.indexBuffer = indices_buffer_.device_ptr + sizeof(unsigned int) * scene_data_.geometries[i].index_offset;
			triangle_input.triangleArray.indexStrideInBytes = sizeof(unsigned int) * 3;
			triangle_input.triangleArray.numIndexTriplets = scene_data_.geometries[i].index_count / 3;

			triangle_input.triangleArray.flags = triangle_input_flags.data();
			triangle_input.triangleArray.numSbtRecords = scene_data_.materials.size();

			triangle_input.triangleArray.sbtIndexOffsetBuffer = material_ids_buffer_.device_ptr + sizeof(unsigned int) * scene_data_.geometries[i].index_offset / 3;
			triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(unsigned int);
			triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(unsigned int);

			OptixAccelBufferSizes gas_buffer_sizes;
			OPTIX_CHECK(optixAccelComputeMemoryUsage(
				optix_context_,
				&accel_options,
				&triangle_input,
				1,
				&gas_buffer_sizes
			));

			CUdeviceptr d_temp_buffer_gas;
			CUDA_CHECK(cudaMalloc(
				reinterpret_cast<void**>(&d_temp_buffer_gas),
				gas_buffer_sizes.tempSizeInBytes
			));

			CUDA_CHECK(cudaMalloc(
				reinterpret_cast<void**>(&d_gas_buffer_[i]),
				gas_buffer_sizes.outputSizeInBytes
			));

			OPTIX_CHECK(optixAccelBuild(
				optix_context_,
				0,                  // CUDA stream
				&accel_options,
				&triangle_input,
				1,                  // num build inputs
				d_temp_buffer_gas,
				gas_buffer_sizes.tempSizeInBytes,
				d_gas_buffer_[i],
				gas_buffer_sizes.outputSizeInBytes,
				&gas_handle_[i],
				nullptr,            // emitted property list
				0                   // num emitted properties
			));

			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
		}

		timer.Stop();

		spdlog::info("GAS Build End : {:05f} ms ", timer.getTimeMS());
	}

	void buildIAS() {
		if (!d_ias_buffer_) {
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_ias_buffer_)));
		}

		Timer timer;
		spdlog::info("IAS Build");
		spdlog::info("IAS Number : {:5d}", scene_data_.instances.size());
		std::vector<OptixInstance> optix_instances(scene_data_.instances.size());

		timer.Start();
		for (int i = 0; i < scene_data_.instances.size(); i++) {
			optix_instances[i].instanceId = i;
			optix_instances[i].sbtOffset = 0;
			optix_instances[i].visibilityMask = 255;
			optix_instances[i].flags = OPTIX_INSTANCE_FLAG_NONE;

			optix_instances[i].traversableHandle = gas_handle_[scene_data_.instances[i].geometry_id];

			Matrix4x3 transforms = transform_matrices_[i];

			float transform[12] =
			{
				transforms.r0.x, transforms.r0.y , transforms.r0.z , transforms.r0.w,
				transforms.r1.x, transforms.r1.y , transforms.r1.z , transforms.r1.w,
				transforms.r2.x, transforms.r2.y , transforms.r2.z , transforms.r2.w
			};

			memcpy(optix_instances[i].transform, transform, sizeof(float) * 12);
		}


		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&d_ias_buffer_),
			optix_instances.size() * sizeof(OptixInstance)
		));

		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_ias_buffer_),
			optix_instances.data(),
			optix_instances.size() * sizeof(OptixInstance),
			cudaMemcpyHostToDevice
		));

		OptixAccelBuildOptions ias_build_options = {};
		ias_build_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
		ias_build_options.operation = OPTIX_BUILD_OPERATION_BUILD;

		OptixBuildInput ias_build_input = {};
		ias_build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
		ias_build_input.instanceArray.instances = d_ias_buffer_;
		ias_build_input.instanceArray.numInstances = optix_instances.size();

		OptixAccelBufferSizes ias_buffer_sizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(
			optix_context_,
			&ias_build_options,
			&ias_build_input,
			1,
			&ias_buffer_sizes
		));

		CUdeviceptr d_temp_buffer_ias;
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&d_temp_buffer_ias),
			ias_buffer_sizes.tempSizeInBytes
		));

		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&d_ias_buffer_),
			ias_buffer_sizes.outputSizeInBytes
		));

		OPTIX_CHECK(optixAccelBuild(
			optix_context_,
			0,                  // CUDA stream
			&ias_build_options,
			&ias_build_input,
			1,                  // num build inputs
			d_temp_buffer_ias,
			ias_buffer_sizes.tempSizeInBytes,
			d_ias_buffer_,
			ias_buffer_sizes.outputSizeInBytes,
			&ias_handle_,
			nullptr,            // emitted property list
			0                   // num emitted properties
		));

		timer.Stop();
		spdlog::info("IAS Build End : {:05f} ms", timer.getTimeMS());

		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_ias)));
	}

	void optixModuleBuild() {
		OptixModuleCompileOptions module_compile_options = {};
		module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
		module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

		pipeline_compile_options_.usesMotionBlur = false;
		pipeline_compile_options_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
		pipeline_compile_options_.numPayloadValues = 3;
		pipeline_compile_options_.numAttributeValues = 3;
		pipeline_compile_options_.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
		pipeline_compile_options_.pipelineLaunchParamsVariableName = "params";
		pipeline_compile_options_.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

		//TODO PTXFileの参照を外部からいじれるようにしておいて
		//Test
		size_t      inputSize = 0;
		const std::vector<char> input = read_file(render_option_.ptxfile_path);
		inputSize = input.size();

		OPTIX_CHECK_LOG(optixModuleCreate(
			optix_context_,
			&module_compile_options,
			&pipeline_compile_options_,
			input.data(),
			inputSize,
			LOG, &LOG_SIZE,
			&optix_module_
		));
	}

	void optixPipelineBuild() {
		OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

		OptixProgramGroupDesc raygen_prog_group_desc = {}; //
		raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		raygen_prog_group_desc.raygen.module = optix_module_;
		raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			optix_context_,
			&raygen_prog_group_desc,
			1,   // num program groups
			&program_group_options,
			LOG, &LOG_SIZE,
			&raygen_prog_group_
		));

		OptixProgramGroupDesc miss_prog_group_desc = {};
		miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		miss_prog_group_desc.miss.module = optix_module_;
		miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";

		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			optix_context_,
			&miss_prog_group_desc,
			1,   // num program groups
			&program_group_options,
			LOG, &LOG_SIZE,
			&miss_prog_group_
		));

		OptixProgramGroupDesc hitgroup_prog_group_desc = {};
		hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		hitgroup_prog_group_desc.hitgroup.moduleCH = optix_module_;
		hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
		hitgroup_prog_group_desc.hitgroup.moduleAH = optix_module_;
		hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ch";
		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			optix_context_,
			&hitgroup_prog_group_desc,
			1,   // num program groups
			&program_group_options,
			LOG, &LOG_SIZE,
			&hitgroup_prog_group_
		));

		memset(&hitgroup_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
		hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		hitgroup_prog_group_desc.hitgroup.moduleCH = optix_module_;
		hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
		hitgroup_prog_group_desc.hitgroup.moduleAH = optix_module_;
		hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__shadow";

		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			optix_context_,
			&hitgroup_prog_group_desc,
			1,   // num program groups
			&program_group_options,
			LOG, &LOG_SIZE,
			&hitgroup_prog_shadow_group_
		));


		const uint32_t    max_trace_depth = 1;
		OptixProgramGroup program_groups[] = { raygen_prog_group_, miss_prog_group_, hitgroup_prog_group_,hitgroup_prog_shadow_group_ };

		OptixPipelineLinkOptions pipeline_link_options = {};
		pipeline_link_options.maxTraceDepth = max_trace_depth;
		OPTIX_CHECK_LOG(optixPipelineCreate(
			optix_context_,
			&pipeline_compile_options_,
			&pipeline_link_options,
			program_groups,
			sizeof(program_groups) / sizeof(program_groups[0]),
			LOG, &LOG_SIZE,
			&optix_pipeline_
		));

		OptixStackSizes stack_sizes = {};
		for (auto& prog_group : program_groups)
		{
			OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, optix_pipeline_));
		}

		uint32_t direct_callable_stack_size_from_traversal;
		uint32_t direct_callable_stack_size_from_state;
		uint32_t continuation_stack_size;
		OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
			0,  // maxCCDepth
			0,  // maxDCDEpth
			&direct_callable_stack_size_from_traversal,
			&direct_callable_stack_size_from_state, &continuation_stack_size));
		OPTIX_CHECK(optixPipelineSetStackSize(optix_pipeline_, direct_callable_stack_size_from_traversal,
			direct_callable_stack_size_from_state, continuation_stack_size,
			1  // maxTraversableDepth
		));
	}

	void optixSBTBuild() {
		CUdeviceptr  raygen_record;
		const size_t raygen_record_size = sizeof(RayGenSbtRecord);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));

		RayGenSbtRecord rg_sbt;
		OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group_, &rg_sbt));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(raygen_record),
			&rg_sbt,
			raygen_record_size,
			cudaMemcpyHostToDevice
		));

		CUdeviceptr miss_record;
		size_t      miss_record_size = sizeof(MissSbtRecord);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
		MissSbtRecord ms_sbt;
		ms_sbt.data = { 0.0f, 0.0f, 0.0f };
		OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group_, &ms_sbt));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(miss_record),
			&ms_sbt,
			miss_record_size,
			cudaMemcpyHostToDevice
		));

		const unsigned int MATCOUNT = scene_data_.materials.size();

		CUdeviceptr hitgroup_record;
		size_t      hitgroup_record_size = sizeof(HitGroupSbtRecord) * RAYTYPE_ * MATCOUNT;

		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));

		std::vector<HitGroupSbtRecord> hg_sbts(RAYTYPE_ * MATCOUNT);
		for (int i = 0; i < MATCOUNT; i++) {
			{
				unsigned int sbt_idx = i * RAYTYPE_;

				hg_sbts[sbt_idx].data.basecolor = scene_data_.materials[i].base_color;
				hg_sbts[sbt_idx].data.basecolor_tex = scene_data_.materials[i].base_color_tex;

				hg_sbts[sbt_idx].data.emmision = scene_data_.materials[i].emmision_color;
				hg_sbts[sbt_idx].data.emmision_tex = scene_data_.materials[i].emmision_color_tex;
				hg_sbts[sbt_idx].data.is_light = scene_data_.materials[i].is_light;

				hg_sbts[sbt_idx].data.metallic = scene_data_.materials[i].metallic;
				hg_sbts[sbt_idx].data.metallic_tex = scene_data_.materials[i].metallic_tex;

				hg_sbts[sbt_idx].data.roughness = scene_data_.materials[i].roughness;
				hg_sbts[sbt_idx].data.roughness_tex = scene_data_.materials[i].roughness_tex;

				hg_sbts[sbt_idx].data.specular = scene_data_.materials[i].specular;

				hg_sbts[sbt_idx].data.sheen = scene_data_.materials[i].sheen;

				hg_sbts[sbt_idx].data.clearcoat = scene_data_.materials[i].clearcoat;
				hg_sbts[sbt_idx].data.clearcoat_tex = scene_data_.materials[i].clearcoat_tex;

				hg_sbts[sbt_idx].data.bump_tex = scene_data_.materials[i].bump_tex;
				hg_sbts[sbt_idx].data.normal_tex = scene_data_.materials[i].normal_tex;

				hg_sbts[sbt_idx].data.transmission = scene_data_.materials[i].transmission;
				hg_sbts[sbt_idx].data.ior = scene_data_.materials[i].ior;

				hg_sbts[sbt_idx].data.ideal_specular = scene_data_.materials[i].ideal_specular;

				hg_sbts[sbt_idx].data.is_thinfilm = scene_data_.materials[i].is_thinfilm;

				OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group_, &hg_sbts[sbt_idx]));
			}

			{
				unsigned int sbt_idx = i * RAYTYPE_ + 1;

				hg_sbts[sbt_idx].data.basecolor = scene_data_.materials[i].base_color;
				hg_sbts[sbt_idx].data.basecolor_tex = scene_data_.materials[i].base_color_tex;

				hg_sbts[sbt_idx].data.emmision = scene_data_.materials[i].emmision_color;
				hg_sbts[sbt_idx].data.emmision_tex = scene_data_.materials[i].emmision_color_tex;
				hg_sbts[sbt_idx].data.is_light = scene_data_.materials[i].is_light;

				hg_sbts[sbt_idx].data.metallic = scene_data_.materials[i].metallic;
				hg_sbts[sbt_idx].data.metallic_tex = scene_data_.materials[i].metallic_tex;

				hg_sbts[sbt_idx].data.roughness = scene_data_.materials[i].roughness;
				hg_sbts[sbt_idx].data.roughness_tex = scene_data_.materials[i].roughness_tex;

				hg_sbts[sbt_idx].data.specular = scene_data_.materials[i].specular;

				hg_sbts[sbt_idx].data.sheen = scene_data_.materials[i].sheen;

				hg_sbts[sbt_idx].data.clearcoat = scene_data_.materials[i].clearcoat;
				hg_sbts[sbt_idx].data.clearcoat_tex = scene_data_.materials[i].clearcoat_tex;

				hg_sbts[sbt_idx].data.bump_tex = scene_data_.materials[i].bump_tex;
				hg_sbts[sbt_idx].data.normal_tex = scene_data_.materials[i].normal_tex;

				hg_sbts[sbt_idx].data.transmission = scene_data_.materials[i].transmission;
				hg_sbts[sbt_idx].data.ior = scene_data_.materials[i].ior;

				hg_sbts[sbt_idx].data.ideal_specular = scene_data_.materials[i].ideal_specular;
				OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_shadow_group_, &hg_sbts[sbt_idx]));
			}
		}
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(hitgroup_record),
			hg_sbts.data(),
			hitgroup_record_size,
			cudaMemcpyHostToDevice
		));

		optix_sbt_.raygenRecord = raygen_record;
		optix_sbt_.missRecordBase = miss_record;
		optix_sbt_.missRecordStrideInBytes = sizeof(MissSbtRecord);
		optix_sbt_.missRecordCount = 1;
		optix_sbt_.hitgroupRecordBase = hitgroup_record;
		optix_sbt_.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
		optix_sbt_.hitgroupRecordCount = RAYTYPE_ * MATCOUNT;
	}
	void textureBind() {
		int numTextures = (int)scene_data_.textures.size();

		carray_objects_.resize(numTextures);
		ctexture_objects_.resize(numTextures);

		for (int textureID = 0; textureID < numTextures; textureID++) {
			auto& texture = scene_data_.textures[textureID];
			Log::DebugLog("Texture ID ", textureID);
			Log::DebugLog("Texture ", texture.tex_name);
			Log::DebugLog("Texture Type", texture.tex_Type);
			cudaResourceDesc res_desc = {};

			cudaChannelFormatDesc channel_desc;
			int32_t width = texture.width;
			int32_t height = texture.height;
			int32_t numComponents = 4;
			int32_t pitch = width * numComponents * sizeof(uint8_t);
			channel_desc = cudaCreateChannelDesc<uchar4>();

			cudaArray_t& pixelArray = carray_objects_[textureID];
			CUDA_CHECK(cudaMallocArray(&pixelArray,
				&channel_desc,
				width, height));

			CUDA_CHECK(cudaMemcpy2DToArray(pixelArray,
				/* offset */0, 0,
				texture.pixel,
				pitch, pitch, height,
				cudaMemcpyHostToDevice));

			res_desc.resType = cudaResourceTypeArray;
			res_desc.res.array.array = pixelArray;

			cudaTextureDesc tex_desc = {};
			tex_desc.addressMode[0] = cudaAddressModeWrap;
			tex_desc.addressMode[1] = cudaAddressModeWrap;
			tex_desc.filterMode = cudaFilterModeLinear;
			tex_desc.readMode = cudaReadModeNormalizedFloat;
			tex_desc.normalizedCoords = 1;
			tex_desc.maxAnisotropy = 1;
			tex_desc.maxMipmapLevelClamp = 99;
			tex_desc.minMipmapLevelClamp = 0;
			tex_desc.mipmapFilterMode = cudaFilterModePoint;
			tex_desc.borderColor[0] = 1.0f;
			tex_desc.sRGB = 1; //png Convert sRGB

			if (texture.tex_Type == TexType::NonColor) {
				tex_desc.sRGB = 0;
			}

			// Create texture object
			cudaTextureObject_t cuda_tex = 0;
			CUDA_CHECK(cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
			ctexture_objects_[textureID] = cuda_tex;
		}
		d_texture_objects_.cpyHostToDevice(ctexture_objects_);
		Log::DebugLog("Textures Loaded");


	}

	void setSky() {
		Log::DebugLog("IBL texture Load");
		if (render_option_.use_IBL) {
			hdrtexture_ = std::make_shared<HDRTexture>(render_option_.IBL_path, render_option_.scene_sky_default);
		}
		else {
			hdrtexture_ = std::make_shared<HDRTexture>(render_option_.scene_sky_default);
		}

		Log::DebugLog("IBL texture Binding");
		{
			auto texture = hdrtexture_;
			cudaResourceDesc res_desc = {};

			cudaChannelFormatDesc channel_desc;
			int32_t width = texture->width;
			int32_t height = texture->height;
			int32_t pitch = width * sizeof(float4);
			channel_desc = cudaCreateChannelDesc<float4>();

			CUDA_CHECK(cudaMallocArray(&ibl_texture_array_,
				&channel_desc,
				width, height));

			CUDA_CHECK(cudaMemcpy2DToArray(ibl_texture_array_,
				0, 0,
				texture->pixel,
				pitch, pitch, height,
				cudaMemcpyHostToDevice));

			res_desc.resType = cudaResourceTypeArray;
			res_desc.res.array.array = ibl_texture_array_;

			cudaTextureDesc tex_desc = {};
			tex_desc.addressMode[0] = cudaAddressModeWrap;
			tex_desc.addressMode[1] = cudaAddressModeWrap;
			tex_desc.filterMode = cudaFilterModeLinear;
			tex_desc.readMode = cudaReadModeElementType;
			tex_desc.normalizedCoords = 1;
			tex_desc.maxAnisotropy = 1;
			tex_desc.maxMipmapLevelClamp = 99;
			tex_desc.minMipmapLevelClamp = 0;
			tex_desc.mipmapFilterMode = cudaFilterModePoint;
			tex_desc.borderColor[0] = 1.0f;
			tex_desc.sRGB = 0;

			CUDA_CHECK(cudaCreateTextureObject(&ibl_texture_object_, &res_desc, &tex_desc, nullptr));
			spdlog::info("IBL Texture Binded");
		}
	}

	//LUT texture bind
	void setLUT() {
		Log::DebugLog("LUT texture Load");
		{
			lut_texture_ = std::make_shared<Texture>(render_option_.LUT_path,TexType::NonColor);

			cudaResourceDesc res_desc = {};

			cudaChannelFormatDesc channel_desc;
			int32_t width = lut_texture_->width;
			int32_t height = lut_texture_->height;
			int32_t numComponents = 4;
			int32_t pitch = width * numComponents * sizeof(uint8_t);
			channel_desc = cudaCreateChannelDesc<uchar4>();

			cudaArray_t& pixelArray = lut_texture_array_;
			CUDA_CHECK(cudaMallocArray(&pixelArray,
				&channel_desc,
				width, height));

			CUDA_CHECK(cudaMemcpy2DToArray(pixelArray,
				/* offset */0, 0,
				lut_texture_->pixel,
				pitch, pitch, height,
				cudaMemcpyHostToDevice));

			res_desc.resType = cudaResourceTypeArray;
			res_desc.res.array.array = pixelArray;

			cudaTextureDesc tex_desc = {};
			tex_desc.addressMode[0] = cudaAddressModeWrap;
			tex_desc.addressMode[1] = cudaAddressModeWrap;
			tex_desc.filterMode = cudaFilterModeLinear;
			tex_desc.readMode = cudaReadModeNormalizedFloat;
			tex_desc.normalizedCoords = 1;
			tex_desc.maxAnisotropy = 1;
			tex_desc.maxMipmapLevelClamp = 99;
			tex_desc.minMipmapLevelClamp = 0;
			tex_desc.mipmapFilterMode = cudaFilterModePoint;
			tex_desc.borderColor[0] = 1.0f;
			tex_desc.sRGB = 0; //png Convert sRGB

			CUDA_CHECK(cudaCreateTextureObject(&lut_texture_object_, &res_desc, &tex_desc, nullptr));
			spdlog::info("LUT Texture Binded");
		}
	}

public:
	Renderer() {

	}

	~Renderer() {
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(optix_sbt_.raygenRecord)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(optix_sbt_.missRecordBase)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(optix_sbt_.hitgroupRecordBase)));

		for (int i = 0; i < d_gas_buffer_.size(); i++) {
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas_buffer_[i])));
		}

		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_ias_buffer_)));

		//for (auto& ctex : ctexture_objects_) {
		//	CUDA_CHECK(cudaDestroyTextureObject(ctex));
		//}

		//for (auto& carray : carray_objects_) {
		//	CUDA_CHECK(cudaFreeArray(carray));
		//}

		//IBL
		CUDA_CHECK(cudaDestroyTextureObject(ibl_texture_object_));
		CUDA_CHECK(cudaFreeArray(ibl_texture_array_));

		//LUT Texture
		CUDA_CHECK(cudaDestroyTextureObject(lut_texture_object_));
		CUDA_CHECK(cudaFreeArray(lut_texture_array_));

		OPTIX_CHECK(optixPipelineDestroy(optix_pipeline_));
		OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_group_));
		OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_shadow_group_));
		OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group_));
		OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group_));
		OPTIX_CHECK(optixModuleDestroy(optix_module_));

		OPTIX_CHECK(optixDeviceContextDestroy(optix_context_));
	}

	void testGeometry() {
		scene_data_.vertices = {
			{0.5f,0.5f,0.0f},
			{0.5f,-0.5f,0.0f},
			{-0.5f,0.5f,0.0f},
			{-0.5f,-0.5f,0.0f},
		};

		scene_data_.indices = {
			0, 1, 2,
			1, 3, 2,
		};

		GeometryData geometry_data;
		geometry_data.index_count = 3;
		geometry_data.index_offset = 0;

		GeometryData geometry_data2;
		geometry_data2.index_count = 3;
		geometry_data2.index_offset = 3;

		scene_data_.geometries.push_back(geometry_data);
		scene_data_.geometries.push_back(geometry_data2);

		InstanceData intance_data;
		intance_data.geometry_id = 0;

		InstanceData intance_data1;
		intance_data1.geometry_id = 1;

		scene_data_.instances.push_back(
			intance_data
		);
		scene_data_.instances.push_back(
			intance_data1
		);
	}

	void setRenderOption(const RenderOption& render_option) {
		render_option_ = render_option;
	}


	void setSceneData(const SceneData& scene_data) {
		scene_data_ = scene_data;
	}

	void loadObjFile(const std::string& filepath, const std::string& filename) {
		Log::StartLog("Load Obj file");
		spdlog::info("loading obj file : {}{}", filepath, filename);

		Timer timer;
		timer.Start();
		if (!loadObj(filepath, filename, scene_data_)) {
			spdlog::warn("Faild loading obj file : {}{}", filepath, filename);
		}
		timer.Stop();

		spdlog::info("Loading Time {:05f}", timer.getTimeMS());
		spdlog::info("Finished loading obj file: {}{}", filepath, filename);
		Log::EndLog("Load Obj file");
	}

	bool loadGLTFfile(const std::string& filepath, const std::string& filename) {
		if (!gltfloader(filepath, filename, scene_data_, render_option_)) {
			spdlog::warn("Faild loading gltf file : {}{}", filepath, filename);
			return false;
		}

		spdlog::info("load gltf file {}{}", filepath, filename);
		return true;
	}

	void build() {
		Log::StartLog("SceneData Copy to Device Memory");
		cpySceneDataToDevice();
		Log::EndLog("SceneData Copy to Device Memory");

		Log::StartLog("Context Initialize");
		optixDeviceContextInitialize();
		Log::EndLog("Context Initialize");

		Log::StartLog("Traversal Handle Build");
		optixTraversalBuild();
		Log::EndLog("Traversal Handle Build");

		Log::StartLog("Module Build");
		optixModuleBuild();
		Log::EndLog("Module Build");

		Log::StartLog("Pipeline Build");
		optixPipelineBuild();
		Log::EndLog("Pipeline Build");

		Log::StartLog("SBT Build");
		optixSBTBuild();
		Log::EndLog("SBT Build");
	}

	bool loadRenderOption(const std::string& filepath) {
		spdlog::info("Load render option file : {}", filepath);

		if (!load_json(filepath, render_option_)) {
			spdlog::error("file load error : {}{}", filepath);
			return false;
		}

		spdlog::info("Success! Loading render option file : {}", filepath);
		return true;
	}

	bool initializeAndRender(const std::string& render_option_path) {
		Timer OverallTimer;

		OverallTimer.Start();
		//Initialize
		{
			if (!loadRenderOption(render_option_path))
			{
				spdlog::error("Faild loading render option file : {}", render_option_path);
				return false;
			}

			if (!loadGLTFfile(render_option_.gltf_path, render_option_.gltf_name)) {
				spdlog::error("Faild loading gltf file : {}{}", render_option_.gltf_path, render_option_.gltf_name);
				return false;
			}

			build();
		}
		OverallTimer.Stop();	
		spdlog::info("Initializing finished : {:4f} ms / {:4f} ms",OverallTimer.getTimeS(),render_option_.time_limit * 60.0f);
		
		Log::StartLog("Render");

		CUstream stream;
		CUDA_CHECK(cudaStreamCreate(&stream));

		Params params;
		CUdeviceptr d_param;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));

		std::string data = "";
		if (render_option_.use_date) {
			data = "data";
		}
		
		//Image Scale Setting
		unsigned int input_image_width = render_option_.image_width;
		unsigned int input_image_height = render_option_.image_height;

		unsigned int output_image_height = render_option_.image_height;
		unsigned int output_image_width = render_option_.image_width;

		if (render_option_.render_mode == RenderMode::DenoiseUpScale2X) {
			input_image_width = render_option_.image_width / 2U;
			input_image_height = render_option_.image_height / 2U;
		}

		//Buffers
		sutil::CUDAOutputBuffer<uchar4> output_buffer(sutil::CUDAOutputBufferType::CUDA_DEVICE, input_image_width, input_image_height);
		BufferObject AOV_Normal(input_image_width, input_image_height);
		BufferObject AOV_Color(input_image_width, input_image_height);
		BufferObject AOV_Albedo(input_image_width, input_image_height);
		BufferObject AOV_Output(output_image_width, output_image_height);
		
		//Denoiser
		DenoiseType denoise_type = DenoiseType::NONE;
		if (render_option_.render_mode == RenderMode::DenoiseUpScale2X) {
			denoise_type = DenoiseType::UPSCALE2X;
		}
		else if (render_option_.render_mode == RenderMode::Default) {
			denoise_type = DenoiseType::NONDENOISE;
		}

		OptixDenoiserManager denosier_manager(
			input_image_width, input_image_height,
			output_image_width, output_image_height,
			optix_context_, stream, denoise_type);

		Timer rendering_timer;
		rendering_timer.Start();
		spdlog::info("Animation Rendering Start");

		for (int frame = render_option_.start_frame; frame < render_option_.end_frame; frame++)
		{
			float time = frame / float(render_option_.fps);
			unsigned int spp = render_option_.max_spp;
			
			//IAS Update
			spdlog::info("IAS Update Start");
			updateIASMatrix(time);
			spdlog::info("IAS Update Finished");

			spdlog::info("Camera Update");
			float3 camera_pos;
			float3 camera_dir;
			float3 camera_up;
			float3 camera_right;
			float camera_f;
			

			//Rendering
			{
				//Camera Update
				camera_f = 2.0 / std::tan(render_option_.camera_fov);

				if (render_option_.camera_animation_id != -1 && render_option_.allow_camera_animation) {
					auto& anim = scene_data_.animations[render_option_.camera_animation_id];
					Affine4x4 affine_pos = anim.getAnimationAffine(time);
					Affine4x4 affine_dir = anim.getRotateAnimationAffine(time);

					float4 trans_camera_pos = affine_pos * make_float4(render_option_.camera_position, 1.0);
					float4 trans_camera_dir = affine_dir * make_float4(render_option_.camera_direction, 0.0);
					float4 trans_camera_up = affine_dir * make_float4(make_float3(0, 1, 0), 0.0);

					camera_pos = make_float3(trans_camera_pos);
					camera_dir = make_float3(trans_camera_dir);
					camera_up = make_float3(trans_camera_up);
					camera_right = normalize(cross(camera_dir, camera_up));
				}
				else {
					camera_pos = render_option_.camera_position;
					camera_dir = render_option_.camera_direction;

					camera_right = cross(camera_dir, make_float3(0, 1, 0));
					camera_up = cross(camera_right, camera_dir);
				}
			}

			spdlog::info("Camera Update Finished");

			//Image infomation
			params.image = output_buffer.map();
			params.image_width = input_image_width;
			params.image_height = input_image_height;

			//Traversal Handle
			params.traversal_handle = ias_handle_;
			
			//Sample
			params.spp = spp;
			params.frame = frame;
			
			//Camera
			params.camera_pos = camera_pos;
			params.camera_dir = camera_dir;
			params.camera_up = camera_up;
			params.camera_right = camera_right;
			params.camera_f = camera_f;

			//Primitive Information
			params.vertices = reinterpret_cast<float3*>(vertices_buffer_.device_ptr);
			params.indices = reinterpret_cast<unsigned int*>(indices_buffer_.device_ptr);
			params.normals = reinterpret_cast<float3*>(normals_buffer_.device_ptr);
			params.texcoords = reinterpret_cast<float2*>(texcoords_buffer_.device_ptr);
			params.colors = reinterpret_cast<float3*>(colors_buffer_.device_ptr);
			params.prim_offsets = reinterpret_cast<unsigned int*>(prim_offset_buffer_.device_ptr);

			//IAS Information
			params.transforms = reinterpret_cast<Matrix4x3*> (transform_matrices_buffer_.device_ptr);
			params.inv_transforms = reinterpret_cast<Matrix4x3*> (inv_transform_matrices_buffer_.device_ptr);
			params.instance_count = scene_data_.instances.size();

			//Textures
			params.textures = reinterpret_cast<cudaTextureObject_t*>(d_texture_objects_.device_ptr);

			//IBL
			params.ibl_texture = ibl_texture_object_;
			params.ibl_intensity = render_option_.IBL_intensity;

			//Light Information
			params.light_prim_ids = reinterpret_cast<unsigned int*>(light_prim_ids_buffer_.device_ptr);
			params.light_prim_count = scene_data_.light_prim_ids.size();
			params.light_prim_emission = reinterpret_cast<float3*>(light_prim_emission_buffer_.device_ptr);
			
			//Raytype
			params.RAYTYPE = RAYTYPE_;

			//AOV
			params.aov_albedo = reinterpret_cast<float4*>(AOV_Albedo.d_gpu_buffer);
			params.aov_color = reinterpret_cast<float4*>(AOV_Color.d_gpu_buffer);
			params.aov_normal = reinterpret_cast<float4*>(AOV_Normal.d_gpu_buffer);

			//LUT
			params.lut_texture = lut_texture_object_;

			CUDA_CHECK(cudaMemcpy(
				reinterpret_cast<void*>(d_param),
				&params, sizeof(params),
				cudaMemcpyHostToDevice
			));

			Timer timer;
			timer.Start();
			spdlog::info("Start render frame {}", frame);
			spdlog::info("Render Info: Image width,height ({},{})", input_image_width, input_image_height);
			spdlog::info("Render Info: Spp {}", params.spp);
			spdlog::info("Render Info: Frame {}", params.frame);
			OPTIX_CHECK(optixLaunch(optix_pipeline_, stream, d_param, sizeof(Params), &optix_sbt_, input_image_width, input_image_height, /*depth=*/1));
			CUDA_SYNC_CHECK();
			
			timer.Stop();
			spdlog::info("End render frame{} : {}s", frame, timer.getTimeS());

			//TimeLimit Check
			//OverallTimer.Stop();	

			//if (OverallTimer.getTimeS() > render_option_.time_limit * 60.0f) {
			//	spdlog::info("Over Time Limit ");
			//	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_param)));
			//	return 0;
			//}

			Timer denoiseTimer;
			denoiseTimer.Start();

			//Denoise
			spdlog::info("Denoising...");
			spdlog::info("Output Resolution : {}x{}", output_image_width, output_image_height);
			{
				denosier_manager.layerSet(
					reinterpret_cast<float4*>(AOV_Albedo.d_gpu_buffer),
					reinterpret_cast<float4*>(AOV_Normal.d_gpu_buffer),
					reinterpret_cast<float4*>(AOV_Color.d_gpu_buffer),
					reinterpret_cast<float4*>(AOV_Output.d_gpu_buffer)
				);

				denosier_manager.denoise();
				CUDA_SYNC_CHECK();
			}
			denoiseTimer.Stop();
			spdlog::info("Denoise Finished! : {} ms", denoiseTimer.getTimeMS());
			
			//Output
			spdlog::info("Export png");
			{
				output_buffer.unmap();
				std::vector<uchar4> bufuchar(output_image_width * output_image_height);
				AOV_Output.cpyGPUBufferToHost();

				float4ConvertColor(AOV_Output.buffer, bufuchar.data(), output_image_width, output_image_height);

				sutil::ImageBuffer buffer;
				buffer.data = bufuchar.data();
				buffer.width = output_image_width;
				buffer.height = output_image_height;
				buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

				std::string str_frame = std::to_string(frame);
				if (str_frame.size() < 2) {
					str_frame = "00" + str_frame;
				}
				else if (str_frame.size() < 3) {
					str_frame = "0" + str_frame;
				}

				//本番仕様
				//std::string imagename = str_frame + ".png";
				std::string imagename = render_option_.image_name + "_" + str_frame + ".png";
				sutil::saveImage(imagename.c_str(), buffer, false);
			}

			rendering_timer.Stop();
			spdlog::info("Frame {} Rendering Finished", params.frame);

			OverallTimer.Stop();	
			spdlog::info("TimeLimit : {:4f} ms / {:4f} ms",OverallTimer.getTimeS(),render_option_.time_limit * 60.0f);
		}

		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_param)));

		rendering_timer.Stop();
		spdlog::info("Animation Rendering End : {}ms", rendering_timer.getTimeS());
		Log::EndLog("Render");
	}
};

