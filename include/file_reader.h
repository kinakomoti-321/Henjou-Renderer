#pragma once
#include <filesystem>
#include <vector>
#include <fstream>
#include <iostream>

std::vector<char> read_file(const std::string& filepath) {
	std::ifstream file(filepath,std::ios::ate | std::ios::binary);
	
	if (!file.is_open()) {
		std::cerr << "Failed to open : " + filepath << std::endl;
	}
	
	const size_t file_size = static_cast<size_t>(file.tellg());
	std::vector<char> buffer(file_size);

	file.seekg(0);
	file.read(buffer.data(), file_size);
	file.close();

	std::cout << buffer.size() << std::endl;

	return buffer;
}
