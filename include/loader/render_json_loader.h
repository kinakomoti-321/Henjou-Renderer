#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cmath>

#include <nlohmann/json.hpp>
#include <sutil/sutil.h>

#include <renderer/render_option.h>
#include <spdlog/spdlog.h>

bool fpsLoader(unsigned int& fps, const std::string& path) {
	std::string filename = path;
	try {
		std::ifstream ifs(filename);

		if (ifs.fail()) {
			std::cout << "File " << filename << " not found" << std::endl;
			return false;
		}
		std::string str;
		while (std::getline(ifs, str)) {
			fps = std::stoi(str);
		}
	}
	catch (std::exception& e)
	{
		std::cerr << "Caught exception: " << e.what() << "\n";
		return false;
	}
	return true;
}

//--------------------
// json file format
//-------------------- 
//Image
//-image_width 
//-image_height
//-image_name
//-image_directory
//-max_spp
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
//-allow_camera_animation
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
//-use_IBL
//-IBL_intensity
//-scene_sky_default
//
//Option
//-use_date
//-save_renderOption


bool load_json(const std::string& filepath, RenderOption& render_option) {
	try {
		std::ifstream ifs(filepath);
		std::string jsonstr;

		if (ifs.fail()) {
			std::cout << "File " << filepath << " not found" << std::endl;
			return false;
		}

		std::string str;
		while (std::getline(ifs, str)) {
			jsonstr += str + "\n";
		}

		auto& jsons = nlohmann::json::parse(jsonstr);

		spdlog::info("Image Setting...");

		//Image
		render_option.image_width = jsons["Image"]["image_width"];
		render_option.image_height = jsons["Image"]["image_height"];
		render_option.image_name = jsons["Image"]["image_name"];
		render_option.image_directory = jsons["Image"]["image_directory"];
		render_option.max_spp = jsons["Image"]["max_spp"];

		spdlog::info("Image Width : {}", render_option.image_width);
		spdlog::info("Image Height : {}", render_option.image_height);
		spdlog::info("Image Name : {}", render_option.image_name);
		spdlog::info("Image Directory : {}", render_option.image_directory);

		spdlog::info("Render Mode Setting...");

		//GLTF
		render_option.gltf_path = jsons["GLTF_file"]["gltf_filepath"];
		render_option.gltf_name = jsons["GLTF_file"]["gltf_filename"];

		//RenderMode
		std::string mode = jsons["Render_mode"];
		if (mode == "Default") {
			render_option.render_mode = RenderMode::Default;
			spdlog::info("Render Mode : Default");
		}
		else if (mode == "Denoise") {
			render_option.render_mode = RenderMode::Denoise;
			spdlog::info("Render Mode : Denoise");
		}
		else if (mode == "Debug") {
			render_option.render_mode = RenderMode::Debug;
			spdlog::info("Render Mode : Debug");
		}
		else if (mode == "DenoiseUpScale2X") {
			render_option.render_mode = RenderMode::DenoiseUpScale2X;
			spdlog::info("Render Mode : Denoise Up Scale 2x");
		}
		else {
			render_option.render_mode = RenderMode::Default;
			spdlog::info("Render Mode : Default");
		}

		//Camera
		spdlog::info("Camera Setting...");
		auto camera_position = jsons["Camera"]["camera_position"];
		render_option.camera_position = make_float3(camera_position[0], camera_position[1], camera_position[2]);
		auto camera_direction = jsons["Camera"]["camera_direction"];
		render_option.camera_direction = make_float3(camera_direction[0], camera_direction[1], camera_direction[2]);
		render_option.camera_fov = M_PI * jsons["Camera"]["camera_fov"] / 180.0f;
		render_option.allow_camera_animation = jsons["Camera"]["allow_camera_animation"];

		spdlog::info("Camera Position : ({},{},{})", render_option.camera_position.x, render_option.camera_position.y, render_option.camera_position.z);
		spdlog::info("Camera Direction : ({},{},{})", render_option.camera_direction.x, render_option.camera_direction.y, render_option.camera_direction.z);
		spdlog::info("Camera FOV : {}", render_option.camera_fov);
		spdlog::info("Camera Animation : {}", render_option.allow_camera_animation);

		//PTX_File
		spdlog::info("PTX File Setting...");
		render_option.ptxfile_path = jsons["PTX_File"]["ptxfile_path"];
		spdlog::info("PTX File Path : {}", render_option.ptxfile_path);

		//Animation
		spdlog::info("Animation Setting...");
		render_option.fps = jsons["Animation"]["fps"];
		render_option.start_frame = jsons["Animation"]["start_frame"];
		render_option.end_frame = jsons["Animation"]["end_frame"];
		render_option.time_limit = jsons["Animation"]["time_limit"];

		unsigned int loaded_fps;
		if (fpsLoader(loaded_fps, "./fps.txt")) {
			spdlog::info("FPS File Loaded");
			render_option.fps = loaded_fps;
		}
		else {
			spdlog::info("FPS File Not Found");
		}

		spdlog::info("FPS : {}", render_option.fps);
		spdlog::info("Start Frame : {}", render_option.start_frame);
		spdlog::info("End Frame : {}", render_option.end_frame);
		spdlog::info("Time Limit : {}", render_option.time_limit);

		//Sky
		spdlog::info("Sky Setting...");
		render_option.IBL_path = jsons["Sky"]["IBL_path"];
		spdlog::info("IBL path : {}", render_option.IBL_path);

		render_option.IBL_intensity = jsons["Sky"]["IBL_intensity"];
		spdlog::info("IBL intensity : {}", render_option.IBL_intensity);

		render_option.use_IBL = jsons["Sky"]["use_IBL"];
		spdlog::info("use_IBL : {}", render_option.use_IBL);

		auto scene_sky_default = jsons["Sky"]["scene_sky_default"];
		render_option.scene_sky_default = make_float3(scene_sky_default[0], scene_sky_default[1], scene_sky_default[2]);

		spdlog::info("scene_sky_default : ({},{},{})", render_option.scene_sky_default.x, render_option.scene_sky_default.y, render_option.scene_sky_default.z);

		//Option
		render_option.use_date = jsons["Option"]["use_date"];
		render_option.save_renderOption = jsons["Option"]["save_renderOption"];

		spdlog::info("use_data : {}", render_option.use_date);
		spdlog::info("save_renderOption : {}", render_option.save_renderOption);

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

