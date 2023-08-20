#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

#include <nlohmann/json.hpp>
#include <sutil/sutil.h>

#include <renderer/render_option.h>
//--------------------
// json file format
//-------------------- 
//Image
//-image_width 
//-image_height
//-image_name
//-image_directory
//
//Render_mode
//
//GLTF_file
//-gltf_filepath;
//-gltf_filename;
// 
//Camera
//-camera_position
//-camera_direction
//-camera_fov
//
//PTX_File
//-ptxfile_path
//
//Animation
//-fps
//-start_frame
//-end_frame
//-time_limit
//
//Sky
//-IBL_path
//-IBL_use
//-IBL_Intensity
//-scene_sky_default
//
//Option
//-use_date
//-save_renderOption

bool load_json(const std::string& filepath, const std::string& filename, RenderOption& render_option) {
	try {
		std::ifstream ifs("./renderer_option.json");
		std::string jsonstr;

		if (ifs.fail()) {
			std::cout << "File " << filepath + filename << " not found" << std::endl;
			return false;
		}

		std::string str;
		while (std::getline(ifs, str)) {
			jsonstr += str + "\n";
		}

		auto& jsons = nlohmann::json::parse(jsonstr);
		
		//Image
		render_option.image_width = jsons["Image"]["image_width"];
		render_option.image_height = jsons["Image"]["image_height"];
		render_option.image_name = jsons["Image"]["image_name"];
		render_option.image_directory = jsons["Image"]["image_directory"];
		
		//RenderMode
		std::string mode = jsons["Render_mode"];
		if (mode == "Default") {
			render_option.render_mode = RenderMode::Default;
		}
		else if (mode == "Denoise") {
			render_option.render_mode = RenderMode::Denoise;
		}
		else if (mode == "Debug") {
			render_option.render_mode = RenderMode::Debug;
		}
		else {
			render_option.render_mode = RenderMode::Default;
		}
		
		//Camera
		auto camera_position = jsons["Camera"]["camera_position"];
		render_option.camera_position = make_float3(camera_position[0],camera_position[1],camera_position[2]);
		auto camera_direction = jsons["Camera"]["camera_direction"];
		render_option.camera_direction = make_float3(camera_direction[0],camera_direction[1],camera_direction[2]);
		render_option.camera_fov = jsons["Camera"]["camera_fov"];

		//PTX_File
		render_option.ptxfile_path = jsons["PTX_File"]["ptxfile_path"];

		//Animation
		render_option.fps = jsons["Animation"]["fps"];
		render_option.start_frame = jsons["Animation"]["start_frame"];
		render_option.end_frame = jsons["Animation"]["end_frame"];
		render_option.time_limit = jsons["Animation"]["time_limit"];
		
		//Sky
		render_option.IBL_path = jsons["Sky"]["IBL_path"];
		render_option.IBL_intensity = jsons["Sky"]["IBL_intensity"];
		render_option.use_IBL = jsons["Sky"]["use_IBL"];

		auto scene_sky_default = jsons["Sky"]["scene_sky_default"];
		render_option.scene_sky_default = make_float3(scene_sky_default[0],scene_sky_default[1],scene_sky_default[2]);

		//Option
		render_option.use_data = jsons["Option"]["use_data"];
		render_option.save_renderOption = jsons["Option"]["save_renderOption"];

		if (render_option.save_renderOption) {
			auto now = std::chrono::system_clock::now();
			std::time_t end_time = std::chrono::system_clock::to_time_t(now);
			std::string time = std::ctime(&end_time);

			auto start = std::chrono::system_clock::now();
			time.erase(std::remove(time.begin(), time.end(), ':'), time.end());
			time.erase(std::remove(time.begin(), time.end(), '\n'), time.end());

			std::ofstream file("renderoption" + time + ".json");
			file << jsonstr;
			file.close();

			std::cout << "Scene File Save " << "scene_file" + time + ".json" << std::endl;
		}

	}
	catch (std::exception& e) {
		std::cerr << "Caught exception: " << e.what() << std::endl;
		return false;
	}

	return true;
}
