#pragma once

#include <iostream>
#include <string>
#include <map>

#include <sutil/sutil.h>

namespace Log {
	template <typename T>
	inline void DebugLog(const std::string& debugstr, const T& input) {
		std::cout << debugstr << " : " << input << std::endl;
	}

	template <typename T>
	inline void DebugLog(const T& debugstr) {
		std::cout << debugstr << std::endl;
	}

	inline void StartLog(const std::string& str) {
		std::cout << "--------------------------------------" << std::endl;
		std::cout << str << " start" << std::endl;
		std::cout << "--------------------------------------" << std::endl;
	}

	inline void EndLog(const std::string& str) {
		std::cout << "--------------------------------------" << std::endl;
		std::cout << str << " End" << std::endl;
		std::cout << "--------------------------------------" << std::endl;
	}
}

std::ostream& operator<<(std::ostream& stream, const float4& f)
{
	stream << "(" << f.x << "," << f.y << "," << f.z << "," << f.w << ")";
	return stream;
}
std::ostream& operator<<(std::ostream& stream, const float3& f)
{
	stream << "(" << f.x << "," << f.y << "," << f.z << ")";
	return stream;
}

std::ostream& operator<<(std::ostream& stream, const float2& f)
{
	stream << "(" << f.x << "," << f.y << ")";
	return stream;
}
template <typename T, typename Q>
std::ostream& operator<<(std::ostream& stream, const std::map<T, Q> f)
{
	for (auto& f1 : f) {
		stream << f1.first << "," << " ";
	}
	return stream;
}
