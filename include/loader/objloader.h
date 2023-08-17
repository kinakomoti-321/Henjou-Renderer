#pragma once

#include <algorithm>
#define TINYOBJLOADER_IMPLEMENTATION
#include <external/tinyobjloader/tiny_obj_loader.h>
#include <renderer/scene.h>
#include <loader/texture_load.h>

#include <sutil/sutil.h>

bool loadObj(const std::string& filepath, const std::string& filename, SceneData& scene_data) {
	tinyobj::ObjReaderConfig reader_config;
	reader_config.mtl_search_path = "./"; // Path to material files

	tinyobj::ObjReader reader;

	if (!reader.ParseFromFile(filepath + filename, reader_config)) {
		if (!reader.Error().empty()) {
			std::cerr << "TinyObjReader: " << reader.Error();
		}
		exit(1);
	}

	if (!reader.Warning().empty()) {
		std::cout << "TinyObjReader: " << reader.Warning();
	}

	auto& attrib = reader.GetAttrib();
	auto& shapes = reader.GetShapes();
	auto& materials = reader.GetMaterials();

	std::cout << "# of vertices  : " << attrib.vertices.size() / 3 << std::endl;
	std::cout << "# of normals   : " << attrib.normals.size() / 3 << std::endl;
	std::cout << "# of texcoords : " << attrib.texcoords.size() / 2 << std::endl;
	std::cout << "# of materials : " << materials.size() << std::endl;
	std::cout << "# of shapes    : " << shapes.size() << std::endl;
    size_t index_offset_ = 0;

    bool has_material = materials.size() > 0;

    std::map<std::string, int> known_tex;

    if (has_material) {
        for (int i = 0; i < materials.size(); i++) {
            //Matrial setting
            Material mat;
            mat.material_name = materials[i].name;

            //Diffuse Color
            mat.base_color = { materials[i].diffuse[0],materials[i].diffuse[1],materials[i].diffuse[2] };

            //Metallic
            mat.metallic = materials[i].metallic;

            //Roghness
            mat.roughness = materials[i].roughness;

            //Sheen
            mat.sheen = materials[i].sheen;

            //Subsurface
            //ClearCoutRoughnessをsubsurfaceとして扱う
            mat.subsurface = materials[i].clearcoat_roughness;
            mat.subsurface_tex = -1;

            //Clearcoat
            mat.clearcoat = materials[i].clearcoat_thickness;
            mat.clearcoat_tex = -1;

            //IOR
            mat.ior = materials[i].ior;

            //Specular
            mat.specular = { materials[i].specular[0],materials[i].specular[1],materials[i].specular[2] };

            //Bump map

            //Emmision
            mat.emmision_color = { materials[i].emission[0],materials[i].emission[1],materials[i].emission[2] };
            mat.emmision_color_tex = -1;

            Log::DebugLog(mat);
            scene_data.materials.push_back(mat);
        }
    }
    else {
        Material mat;
        mat.base_color = { 1,1,1 };
        mat.emmision_color = { 0,0,0 };
        scene_data.materials.push_back(mat);
    }


    //シェイプ数分のループ
    for (size_t s = 0; s < shapes.size(); s++) {
        size_t index_offset = 0;
        GeometryData geo_data;
        geo_data.index_offset = index_offset_;
        //シェイプのフェイス分のループ
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            //シェイプsの面fに含まれる頂点数
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
            float3 nv[3];
            for (size_t v = 0; v < fv; v++) {
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                //シェイプのv番目の頂点座標
                tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

                scene_data.vertices.push_back({ vx ,vy, vz });

                if (idx.normal_index >= 0) {
                    //シェイプのv番目の法線ベクトル
                    tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                    tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                    tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];

                    scene_data.normals.push_back({ nx,ny,nz });
                }
                else {
                    nv[v] = float3{ vx, vy, vz };
                }
                if (idx.texcoord_index >= 0) {
                    //シェイプのv番目のUV
                    tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                    tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];

                    // std::cout << f << std::endl;
                    scene_data.texcoords.push_back({ tx, ty });
                }
                else {
                    scene_data.texcoords.push_back({ 0, 0 });
                }
                //v番目のindex
                scene_data.indices.push_back(static_cast<unsigned int>(index_offset_ + index_offset + v));
            }

            if (attrib.normals.size() == 0) {
                const float3 nv1 = normalize(nv[1] - nv[0]);
                const float3 nv2 = normalize(nv[2] - nv[0]);
                float3 geoNormal = normalize(cross(nv1, nv2));
                scene_data.normals.push_back(geoNormal);
                scene_data.normals.push_back(geoNormal);
                scene_data.normals.push_back(geoNormal);
            }

            if (has_material) {
                scene_data.material_ids.push_back(shapes[s].mesh.material_ids[f]);
            }
            else {
                scene_data.material_ids.push_back(0);
            }
            index_offset += fv;

        }
        geo_data.index_count = index_offset;
        InstanceData ins_data;
        ins_data.geometry_id = s;
        index_offset_ += index_offset;

        scene_data.geometries.push_back(geo_data);
        scene_data.instances.push_back(ins_data);
    }

}
