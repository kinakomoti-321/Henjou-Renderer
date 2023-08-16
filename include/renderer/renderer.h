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

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;

void configureCamera( sutil::Camera& cam, const uint32_t width, const uint32_t height )
{
    cam.setEye( {0.0f, 0.0f, 2.0f} );
    cam.setLookat( {0.0f, 0.0f, 0.0f} );
    cam.setUp( {0.0f, 1.0f, 3.0f} );
    cam.setFovY( 45.0f );
    cam.setAspectRatio( (float)width / (float)height );
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

struct IASData {

};

enum RenderMode {
	Default, //
	Denoise, //Denoised image output
	Debug  //Position,BaseColor,Normal,Texcoord image output
};

struct RenderOption {
	unsigned int image_width = 512;
	unsigned int image_height = 512;

	bool is_animation;
	unsigned int fps;
	unsigned int start_frame;
	unsigned int end_frame;

	float camera_fov;
	unsigned int camera_animation_id;

	RenderMode render_mode = Default;
		
};

struct GeometryData {

};

struct InstanceData {

};

struct AnimationData {

};

struct SceneData {
	std::vector<float3> vertices;
	std::vector<unsigned int> indices;
	std::vector<float3> normals;
	std::vector<float2> texcoords;
	std::vector<float3> colors;
	std::vector<unsigned int> material_ids;

};

class Renderer {
	RenderOption render_option_;
	SceneData scene_data_;

	OptixDeviceContext optix_context_ = nullptr;

	OptixModule optix_module_ = nullptr;

	OptixPipeline optix_pipeline_ = nullptr;
	OptixPipelineCompileOptions pipeline_compile_options_ = {};

	OptixProgramGroup raygen_prog_group_ = nullptr;
	OptixProgramGroup miss_prog_group_ = nullptr;
	OptixProgramGroup hitgroup_prog_group_ = nullptr;
	OptixProgramGroup any_prog_group_ = nullptr;

	OptixShaderBindingTable optix_sbt_ = {};

	OptixTraversableHandle gas_handle;
	CUdeviceptr            d_gas_output_buffer;



private:
	void optixDeviceContextInitialize() {
		CUDA_CHECK(cudaFree(0));

		OPTIX_CHECK(optixInit());

		OptixDeviceContextOptions options = {};
		options.logCallbackFunction = &context_log_cb;
		options.logCallbackLevel = 4;

		CUcontext cuCtx = 0;  // zero means take the current context
		OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &optix_context_));
	}

	void optixTraversalBuild() {
		OptixAccelBuildOptions accel_options = {};
		accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
		accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

		const std::array<float3, 3> vertices =
		{ {
			  { -0.5f, -0.5f, 0.0f },
			  {  0.5f, -0.5f, 0.0f },
			  {  0.0f,  0.5f, 0.0f }
		} };

		const size_t vertices_size = sizeof( float3 )*vertices.size();
		CUdeviceptr d_vertices=0;
		CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_vertices ), vertices_size ) );
		CUDA_CHECK( cudaMemcpy(
					reinterpret_cast<void*>( d_vertices ),
					vertices.data(),
					vertices_size,
					cudaMemcpyHostToDevice
					) );

		const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
		OptixBuildInput triangle_input = {};
		triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
		triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
		triangle_input.triangleArray.numVertices   = static_cast<uint32_t>( vertices.size() );
		triangle_input.triangleArray.vertexBuffers = &d_vertices;
		triangle_input.triangleArray.flags         = triangle_input_flags;
		triangle_input.triangleArray.numSbtRecords = 1;

		OptixAccelBufferSizes gas_buffer_sizes;
		OPTIX_CHECK( optixAccelComputeMemoryUsage(
					optix_context_,
					&accel_options,
					&triangle_input,
					1, // Number of build inputs
					&gas_buffer_sizes
					) );
		CUdeviceptr d_temp_buffer_gas;
		CUDA_CHECK( cudaMalloc(
					reinterpret_cast<void**>( &d_temp_buffer_gas ),
					gas_buffer_sizes.tempSizeInBytes
					) );
		CUDA_CHECK( cudaMalloc(
					reinterpret_cast<void**>( &d_gas_output_buffer ),
					gas_buffer_sizes.outputSizeInBytes
					) );

		OPTIX_CHECK( optixAccelBuild(
					optix_context_,
					0,                  // CUDA stream
					&accel_options,
					&triangle_input,
					1,                  // num build inputs
					d_temp_buffer_gas,
					gas_buffer_sizes.tempSizeInBytes,
					d_gas_output_buffer,
					gas_buffer_sizes.outputSizeInBytes,
					&gas_handle,
					nullptr,            // emitted property list
					0                   // num emitted properties
					) );

		CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_gas ) ) );
		CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_vertices        ) ) );
	}

	void optixModuleBuild() {
		OptixModuleCompileOptions module_compile_options = {};
		module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
		module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

		pipeline_compile_options_.usesMotionBlur        = false;
		pipeline_compile_options_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
		pipeline_compile_options_.numPayloadValues      = 3;
		pipeline_compile_options_.numAttributeValues    = 3;
		pipeline_compile_options_.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
		pipeline_compile_options_.pipelineLaunchParamsVariableName = "params";
		pipeline_compile_options_.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

		//const char* input      = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "henjouRendererCU.cu", inputSize );

		//TODO PTXFileÇÃéQè∆ÇäOïîÇ©ÇÁÇ¢Ç∂ÇÍÇÈÇÊÇ§Ç…ÇµÇƒÇ®Ç¢Çƒ
		size_t      inputSize  = 0;
		const std::vector<char> input = read_file("C:\\Users\\PC\\Documents\\Optix\\build\\lib\\ptx\\Debug\\HenjouRenderer_generated_henjouRendererCU.cu.optixir");
		inputSize = input.size();

		OPTIX_CHECK_LOG( optixModuleCreate(
					optix_context_,
					&module_compile_options,
					&pipeline_compile_options_,
					input.data(),
					inputSize,
					LOG, &LOG_SIZE,
					&optix_module_
					) );
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
		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			optix_context_,
			&hitgroup_prog_group_desc,
			1,   // num program groups
			&program_group_options,
			LOG, &LOG_SIZE,
			&hitgroup_prog_group_
		));


		const uint32_t    max_trace_depth  = 1;
		OptixProgramGroup program_groups[] = { raygen_prog_group_, miss_prog_group_, hitgroup_prog_group_ };

		OptixPipelineLinkOptions pipeline_link_options = {};
		pipeline_link_options.maxTraceDepth            = max_trace_depth;
		OPTIX_CHECK_LOG( optixPipelineCreate(
					optix_context_,
					&pipeline_compile_options_,
					&pipeline_link_options,
					program_groups,
					sizeof( program_groups ) / sizeof( program_groups[0] ),
					LOG, &LOG_SIZE,
					&optix_pipeline_
					) );

		OptixStackSizes stack_sizes = {};
		for( auto& prog_group : program_groups )
		{
			OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes, optix_pipeline_ ) );
		}

		uint32_t direct_callable_stack_size_from_traversal;
		uint32_t direct_callable_stack_size_from_state;
		uint32_t continuation_stack_size;
		OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace_depth,
												 0,  // maxCCDepth
												 0,  // maxDCDEpth
												 &direct_callable_stack_size_from_traversal,
												 &direct_callable_stack_size_from_state, &continuation_stack_size ) );
		OPTIX_CHECK( optixPipelineSetStackSize( optix_pipeline_, direct_callable_stack_size_from_traversal,
												direct_callable_stack_size_from_state, continuation_stack_size,
												1  // maxTraversableDepth
												) );
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

		CUdeviceptr hitgroup_record;
		size_t      hitgroup_record_size = sizeof(HitGroupSbtRecord);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));
		HitGroupSbtRecord hg_sbt;
		OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group_, &hg_sbt));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(hitgroup_record),
			&hg_sbt,
			hitgroup_record_size,
			cudaMemcpyHostToDevice
		));

		optix_sbt_.raygenRecord = raygen_record;
		optix_sbt_.missRecordBase = miss_record;
		optix_sbt_.missRecordStrideInBytes = sizeof(MissSbtRecord);
		optix_sbt_.missRecordCount = 1;
		optix_sbt_.hitgroupRecordBase = hitgroup_record;
		optix_sbt_.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
		optix_sbt_.hitgroupRecordCount = 1;
	}
	
public: 
	Renderer() {

	}
	~Renderer() {
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(optix_sbt_.raygenRecord)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(optix_sbt_.missRecordBase)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(optix_sbt_.hitgroupRecordBase)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas_output_buffer)));

		OPTIX_CHECK(optixPipelineDestroy(optix_pipeline_));
		OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_group_));
		OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group_));
		OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group_));
		OPTIX_CHECK(optixModuleDestroy(optix_module_));

		OPTIX_CHECK(optixDeviceContextDestroy(optix_context_));
	}

	void setRenderOption(const RenderOption& render_option) {
		render_option_ = render_option;	
	}

	void setSceneData(const SceneData& scene_data) {
		scene_data_ = scene_data;
	}
	void build() {
		optixDeviceContextInitialize();
		optixTraversalBuild();
		optixModuleBuild();
		optixPipelineBuild();
		optixSBTBuild();
	}
	void render() {
        sutil::CUDAOutputBuffer<uchar4> output_buffer( sutil::CUDAOutputBufferType::CUDA_DEVICE, render_option_.image_width , render_option_.image_height );

        {
            CUstream stream;
            CUDA_CHECK( cudaStreamCreate( &stream ) );

            sutil::Camera cam;
            configureCamera( cam, render_option_.image_width, render_option_.image_height );

            Params params;
            params.image        = output_buffer.map();
            params.image_width  = render_option_.image_width;
            params.image_height = render_option_.image_height;
            params.traversal_handle = gas_handle;
            params.cam_eye      = cam.eye();
            cam.UVWFrame( params.cam_u, params.cam_v, params.cam_w );

            CUdeviceptr d_param;
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( Params ) ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( d_param ),
                        &params, sizeof( params ),
                        cudaMemcpyHostToDevice
                        ) );

            OPTIX_CHECK( optixLaunch( optix_pipeline_, stream, d_param, sizeof( Params ), &optix_sbt_, render_option_.image_width, render_option_.image_height, /*depth=*/1 ) );
            CUDA_SYNC_CHECK();

            output_buffer.unmap();
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_param ) ) );
        }

        {
            sutil::ImageBuffer buffer;
            buffer.data         = output_buffer.getHostPointer();
            buffer.width        = render_option_.image_width;
            buffer.height       = render_option_.image_height;
            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
            sutil::displayBufferWindow( "test" , buffer);
        }
	}
};

