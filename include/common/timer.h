#pragma once

#include <chrono>

class Timer {
private:
	std::chrono::system_clock::time_point start;
	std::chrono::system_clock::time_point end;

public:
	Timer() {}
	~Timer() {}

	void Start() {
		start = std::chrono::system_clock::now();
	}

	void Stop() {
		end = std::chrono::system_clock::now();
	}

	double getTimeS() {
		std::chrono::duration<double> elapsed_seconds = end - start;
		return elapsed_seconds.count();
	}

	double getTimeMS() {
		std::chrono::duration<double, std::milli> elapsed_milliseconds = end - start;
		return elapsed_milliseconds.count();
	}

	double getTimeUS() {
		std::chrono::duration<double, std::micro> elapsed_microseconds = end - start;
		return elapsed_microseconds.count();
	}

	double getTimeNS() {
		std::chrono::duration<double, std::nano> elapsed_nanoseconds = end - start;
		return elapsed_nanoseconds.count();
	}
};
