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

#include <HenjouRenderer/henjouRenderer.h>
#include <file_reader.h>
#include <renderer/scene.h>
#include <renderer/material.h>
#include <loader/objloader.h>
#include <cu/cuda_buffer.h>
#include <common/log.h>
#include <common/timer.h>

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
	cam.setEye({ 4.0f, 0.0f, 0.0f });
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


enum RenderMode {
	Default, //
	Denoise, //Denoised image output
	Debug  //Position,BaseColor,Normal,Texcoord image output
};

struct RenderOption {
	unsigned int image_width = 1024;
	unsigned int image_height = 1024;

	bool is_animation;
	unsigned int fps;
	unsigned int start_frame;
	unsigned int end_frame;

	float camera_fov;
	unsigned int camera_animation_id;

	RenderMode render_mode = Default;

	std::string ptx_path;
};


struct AnimationData {

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

	std::vector<OptixTraversableHandle> gas_handle_;
	std::vector<CUdeviceptr> d_gas_buffer_;

	const unsigned int RAYTYPE_ = 2;

private:

	void cpySceneDataToDevice() {
		vertices_buffer_.cpyHostToDevice(scene_data_.vertices);
		indices_buffer_.cpyHostToDevice(scene_data_.indices);
		normals_buffer_.cpyHostToDevice(scene_data_.normals);
		texcoords_buffer_.cpyHostToDevice(scene_data_.texcoords);
		material_ids_buffer_.cpyHostToDevice(scene_data_.material_ids);
		colors_buffer_.cpyHostToDevice(scene_data_.colors);
		prim_offset_buffer_.cpyHostToDevice(scene_data_.prim_offset);

		spdlog::info("Scene Data");
		spdlog::info("number of vertex : {:16d}", scene_data_.vertices.size());
		spdlog::info("number of index : {:16d}", scene_data_.indices.size());
		spdlog::info("number of normal : {:16d}", scene_data_.normals.size());
		spdlog::info("number of texcoord : {:16d}", scene_data_.texcoords.size());
		spdlog::info("number of material id : {:16d}", scene_data_.material_ids.size());
		spdlog::info("number of material : {:16d}", scene_data_.materials.size());
		spdlog::info("number of color : {:16d}", scene_data_.colors.size());
		spdlog::info("number of prim offset : {:16d}", scene_data_.prim_offset.size());

		std::cout << scene_data_.material_ids << std::endl;
	}

	void optixDeviceContextInitialize() {
		CUDA_CHECK(cudaFree(0));

		OPTIX_CHECK(optixInit());

		OptixDeviceContextOptions options = {};
		options.logCallbackFunction = &context_log_cb;
		options.logCallbackLevel = 4;

		CUcontext cuCtx = 0;
		OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &optix_context_));

		spdlog::info("Optix Device Context Initialize");
	}

	void optixTraversalBuild() {
		OptixAccelBuildOptions accel_options = {};
		accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
		accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

		gas_handle_.resize(scene_data_.geometries.size());
		d_gas_buffer_.resize(scene_data_.geometries.size());
		const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };

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
			triangle_input.triangleArray.flags = triangle_input_flags;
			triangle_input.triangleArray.numSbtRecords = 1;

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

			float transform[12] = { 1,0,0,0,0,1,0,0,0,0,1,0 };
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

		//TODO PTXFileÇÃéQè∆ÇäOïîÇ©ÇÁÇ¢Ç∂ÇÍÇÈÇÊÇ§Ç…ÇµÇƒÇ®Ç¢Çƒ
		size_t      inputSize = 0;
		const std::vector<char> input = read_file("C:\\Users\\PC\\Documents\\Optix\\build\\lib\\ptx\\Debug\\HenjouRenderer_generated_henjouRendererCU.cu.optixir");
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
		OptixProgramGroup program_groups[] = { raygen_prog_group_, miss_prog_group_, hitgroup_prog_group_,hitgroup_prog_shadow_group_};

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
		ms_sbt.data = { 0.3f, 0.1f, 0.2f };
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

		//HitGroupSbtRecord hg_sbt;
		//OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group_, &hg_sbt));
		//CUDA_CHECK(cudaMemcpy(
		//	reinterpret_cast<void*>(hitgroup_record),
		//	&hg_sbt,
		//	hitgroup_record_size,
		//	cudaMemcpyHostToDevice
		//));

		optix_sbt_.raygenRecord = raygen_record;
		optix_sbt_.missRecordBase = miss_record;
		optix_sbt_.missRecordStrideInBytes = sizeof(MissSbtRecord);
		optix_sbt_.missRecordCount = 1;
		optix_sbt_.hitgroupRecordBase = hitgroup_record;
		optix_sbt_.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
		optix_sbt_.hitgroupRecordCount = RAYTYPE_ * MATCOUNT;
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

	void render() {
		Log::StartLog("Render");
		sutil::CUDAOutputBuffer<uchar4> output_buffer(sutil::CUDAOutputBufferType::CUDA_DEVICE, render_option_.image_width, render_option_.image_height);
		{
			CUstream stream;
			CUDA_CHECK(cudaStreamCreate(&stream));

			sutil::Camera cam;
			configureCamera(cam, render_option_.image_width, render_option_.image_height);

			Params params;
			params.image = output_buffer.map();
			params.image_width = render_option_.image_width;
			params.image_height = render_option_.image_height;
			params.traversal_handle = ias_handle_;

			params.vertices = reinterpret_cast<float3*>(vertices_buffer_.device_ptr);
			params.indices = reinterpret_cast<unsigned int*>(indices_buffer_.device_ptr);
			params.normals = reinterpret_cast<float3*>(normals_buffer_.device_ptr);
			params.texcoords = reinterpret_cast<float2*>(texcoords_buffer_.device_ptr);
			params.colors = reinterpret_cast<float3*>(colors_buffer_.device_ptr);
			params.prim_offsets = reinterpret_cast<unsigned int*>(prim_offset_buffer_.device_ptr);

			params.cam_eye = cam.eye();
			cam.UVWFrame(params.cam_u, params.cam_v, params.cam_w);



			CUdeviceptr d_param;
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));
			CUDA_CHECK(cudaMemcpy(
				reinterpret_cast<void*>(d_param),
				&params, sizeof(params),
				cudaMemcpyHostToDevice
			));

			OPTIX_CHECK(optixLaunch(optix_pipeline_, stream, d_param, sizeof(Params), &optix_sbt_, render_option_.image_width, render_option_.image_height, /*depth=*/1));
			CUDA_SYNC_CHECK();

			output_buffer.unmap();
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_param)));
		}

		{
			sutil::ImageBuffer buffer;
			buffer.data = output_buffer.getHostPointer();
			buffer.width = render_option_.image_width;
			buffer.height = render_option_.image_height;
			buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
			sutil::displayBufferWindow("test", buffer);
		}

		Log::EndLog("Render");
	}
};

