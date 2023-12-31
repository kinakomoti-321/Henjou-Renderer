#pragma once

#include <iostream>
#include <string>

//--------------------
// json file format
//-------------------- 
//image_width 
//image_height
//image_name
//image_directory
//max_spp
//
//GLTF path
//
//render_mode
// 
//camera_position
//camera_direction
//camera_fov
//
//ptxfile_path
//
//fps
//start_frame
//end_frame
//time_limit
//
//IBL_path
//IBL_use
//IBL_Intensity
//scene_sky_default
//
//use_date
//save_renderOption

enum RenderMode {
	Default, //
	Denoise, //Denoised image output
	DenoiseUpScale2X, //Denoised image output and upscale 2x
	Debug  //Position,BaseColor,Normal,Texcoord image output
};

struct RenderOption {
	bool is_set = false;

	unsigned int image_width = 1024;
	unsigned int image_height = 1024;
	std::string image_name = "test";
	std::string image_directory = "./";
	unsigned int max_spp = 100;

	std::string gltf_path = "./";
	std::string gltf_name = "test.gltf";

	unsigned int fps = 24;
	unsigned int start_frame = 0;
	unsigned int end_frame = 1;
	float time_limit = 1.0;

	bool allow_camera_animation = false;
	float camera_fov = 45;
	float3 camera_position = { 0.0,0.0,0.0 };
	float3 camera_direction = { 0.0,0.0,-1.0 };
	int camera_animation_id = -1;

	RenderMode render_mode = Default;

	std::string ptxfile_path;
	
	bool use_IBL = false;
	std::string IBL_path;
	float IBL_intensity;

	float3 scene_sky_default = { 0.0,0.0,0.0 };

	bool use_date = false;
	bool save_renderOption = false;

	//LUT
	std::string LUT_path;

};
