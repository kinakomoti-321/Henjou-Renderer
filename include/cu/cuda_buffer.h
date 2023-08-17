#pragma once

#include <vector>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>

namespace cuh {
	struct CUDevicePointer {

		CUdeviceptr device_ptr = 0;
		size_t size_in_bytes = 0;

		CUDevicePointer() {}
		~CUDevicePointer() {
			if (device_ptr)
				memFree();
			device_ptr = 0;
		}

		CUdeviceptr* getDevicePtr() {
			return &device_ptr;
		}

		void memFree() {
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(device_ptr)));
		}

		template <typename T>
		void cpyHostToDevice(const std::vector<T>& i_data) {
			if (device_ptr) {
				memFree();
			}
			size_in_bytes = sizeof(T) * i_data.size();
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_ptr), size_in_bytes));
			CUDA_CHECK(cudaMemcpy(
				reinterpret_cast<void*>(device_ptr),
				i_data.data(),
				size_in_bytes,
				cudaMemcpyHostToDevice
			));
		}
	};
}
