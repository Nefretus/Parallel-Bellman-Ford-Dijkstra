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
	//for (size_t size = 1000; size <= 20000; size += 1000)
	int vertex_count = 500;
	file << "Ford" << std::endl;
	file << "Parallel [ms]" << ";" << "Seq [ms]" << std::endl;
	run_bellman_ford(vertex_count, file, threads_per_block);
	file << "Dijkstra" << std::endl;
	file << "Parallel [ms]" << ";" << "Seq [ms]" << std::endl;
	run_dijkstra(vertex_count, file, threads_per_block);
	file.close();
	return 0;
}
