#pragma once

void run_dijkstra(size_t size, std::ofstream& output_file, int block_size, bool run_seq = true, bool run_parallel = true);
void run_bellman_ford(size_t size, std::ofstream& output_file, int block_size, bool run_seq = true, bool run_parallel = true);

