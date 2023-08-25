#pragma once

#include <vector>
#include <sutil/sutil.h>
#include <renderer/material.h>
#include <renderer/texture.h>
#include <renderer/animation.h>

struct GeometryData {
	unsigned int index_offset;
	unsigned int index_count;
};

struct InstanceData {
	unsigned int geometry_id;
	unsigned int animation_id;
};

struct SceneData {
	std::vector<float3> vertices;
	std::vector<unsigned int> indices;
	std::vector<unsigned int> material_ids;
	std::vector<float3> normals;
	std::vector<float2> texcoords;
	std::vector<float3> colors;

	std::vector<Material> materials;
	std::vector<Texture> textures;
	std::vector<unsigned int> light_prim_ids;
	std::vector<float3> light_prim_emission;

	std::vector<Animation> animations;
	std::vector<GeometryData> geometries;
	std::vector<InstanceData> instances;
	std::vector<unsigned int> prim_offset;
};

