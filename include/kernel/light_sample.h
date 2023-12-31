#pragma once

#include <cuda/helpers.h>
#include <sutil/vec_math.h>
#include <kernel/cmj.h>
#include <kernel/Payload.h>
#include <kernel/math.h>

__forceinline__ __device__ float3 light_sample(CMJState& state, float& pdf, float3& normal,float3& emission)
{

	if (params.light_prim_count < 1) {
		pdf = -1.0;
		normal = make_float3(0.0f);
		return make_float3(0.0f);
	}

	float p = cmj_1d(state);
	int index = (int)(p * params.light_prim_count);
	if (index == params.light_prim_count) index--;

	unsigned int prim_index = params.light_prim_ids[index];


	//BinarySearch
	unsigned int left = 0U;
	unsigned int right = params.instance_count - 1;
	unsigned int middle = (left + right) / 2U;

	while (left <= right) {
		if (params.prim_offsets[middle] <= prim_index) {
			left = middle + 1;
		}
		else {
			right = middle - 1;
		}
		middle = (left + right) / 2;
	}
	
	float select_pdf = 1.0f / params.light_prim_count;

	unsigned int instance_index = middle;
	const Matrix4x3 transform = params.transforms[instance_index];
	const Matrix4x3 transform_inv = params.inv_transforms[instance_index];

	const unsigned int idx0 = params.indices[prim_index * 3 + 0];
	const unsigned int idx1 = params.indices[prim_index * 3 + 1];
	const unsigned int idx2 = params.indices[prim_index * 3 + 2];

	const float3 v0 = transform_position(transform, params.vertices[idx0]);
	const float3 v1 = transform_position(transform, params.vertices[idx1]);
	const float3 v2 = transform_position(transform, params.vertices[idx2]);

	const float3 n0 = transform_normal(transform_inv,params.normals[idx0]);
	const float3 n1 = transform_normal(transform_inv,params.normals[idx1]);
	const float3 n2 = transform_normal(transform_inv,params.normals[idx2]);

	const float light_area = length(cross(v1 - v0, v2 - v0)) * 0.5f;
	
	float2 xi = cmj_2d(state);
	
	float f1 = 1.0f - sqrt(xi.x);
	float f2 = sqrt(xi.x) * (1.0f - xi.y);
	float f3 = sqrt(xi.x) * xi.y;
	
	const float3 light_position = v0 * f1 + v1 * f2 + v2 * f3;
	const float3 light_normal = normalize(n0 * f1 + n1 * f2 + n2 * f3);

	pdf = 1.0 / light_area;
	pdf *= select_pdf;
	normal = light_normal;
	emission = params.light_prim_emission[index];

	return light_position;
}

__forceinline__ __device__ float getLightPDF(const unsigned int& primID,const unsigned int& instID) {

	const unsigned int idx0 = params.indices[primID * 3 + 0];
	const unsigned int idx1 = params.indices[primID * 3 + 1];
	const unsigned int idx2 = params.indices[primID * 3 + 2];

	const Matrix4x3 transform = params.transforms[instID];

	const float3 v0 = transform_position(transform, params.vertices[idx0]);
	const float3 v1 = transform_position(transform, params.vertices[idx1]);
	const float3 v2 = transform_position(transform, params.vertices[idx2]);

	float light_area = length(cross(v1 - v0, v2 - v0)) * 0.5f;
	
	return 1.0f / (light_area * params.light_prim_count);
}

