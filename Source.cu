#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <random>
#include <chrono>
#include <fstream>
#include "par.cuh"

constexpr int threads_per_block = 256;

int main(void) {
	std::locale pol("pl_PL");
	std::ofstream file("test.csv");
	file.imbue(pol);
	int vertex_count = 5000;
	run_dijkstra(vertex_count, file, threads_per_block);
	file.close();
	return 0;
}
