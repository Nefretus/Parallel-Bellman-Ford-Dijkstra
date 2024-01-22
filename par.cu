#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include "seq.h"

////////////////////////  Bellman Ford parallel ////////////////////////

__global__ void bellman_ford_parallel(int* matrix, int* dist, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int skip = blockDim.x * gridDim.x;
    if (id < n) {
        for (int i = 0; i < n; i++) {
            for (int j = id; j < n; j += skip) {
                if (matrix[i * n + j] != 0 && dist[i] != INT_MAX && dist[i] + matrix[i * n + j] < dist[j])
                    dist[j] = dist[i] + matrix[i * n + j];
            }
        }
    }
}

////////////////////////  Dijkstra parallel ////////////////////////

__global__ void init(int* unvsited, int* frontier_v, int* distances, size_t gSize) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < gSize) {
        unvsited[id] = 1; // is unvisted = false
        frontier_v[id] = 0;
        distances[id] = INT_MAX;
    }
    if (id == 0) {
        distances[id] = 0;
        unvsited[id] = 0;
        frontier_v[id] = 1;
    }
}

__global__ void relax_f(int* c, int* f, int* u, int* matrix, size_t size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        if (f[id]) {
            for (int i = 0; i < size; i++) {
                if (matrix[id * size + i] != 0 && u[i]) {
                    atomicMin(&c[i], c[id] + matrix[id * size + i]);
                }
            }
        }
    }
}

__global__ void update(int* U, int* F, int* c, int* closest, size_t size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        F[id] = 0;
        if ((U[id]) && (c[id] == (*closest))) {
            U[id] = 0;
            F[id] = 1;
        }
    }
}

__global__ void find_min(int *u, int* c, int* minimums, int n) {
    extern __shared__ int sdata[];
    int thid = threadIdx.x;
    int i = blockIdx.x * (2 * blockDim.x) + threadIdx.x;
    int j = i + blockDim.x;
    int data1 = (i < n) ? (u[i] ? c[i] : INT_MAX) : INT_MAX;
    int data2 = (j < n) ? (u[j] ? c[j] : INT_MAX) : INT_MAX;
    sdata[thid] = min(data1, data2);
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (thid < s)
            sdata[thid] = min(sdata[thid], sdata[thid + s]);
        __syncthreads();
    }
    if (thid == 0) 
        minimums[blockIdx.x] = sdata[0];
}

void run_dijkstra(size_t size, std::ofstream& output_file, int block_size) {
    size_t gSize = size;

    int* adjMat;

    int* d_adj_mat;
    int* d_unvisited;
    int* d_frontier;
    int* d_estimates;

    std::vector<std::vector<int>> adj_matrix = generate_adj_matrix(size);
    adjMat = (int*)malloc(size * size * sizeof(int));
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            adjMat[i * size + j] = adj_matrix[i][j];
        }
    }

    int* shortest_out;
    shortest_out = (int*)malloc(sizeof(int) * gSize);

    cudaMalloc((void**)&d_adj_mat, sizeof(int) * gSize * gSize);
    cudaMalloc((void**)&d_unvisited, sizeof(int) * gSize);
    cudaMalloc((void**)&d_frontier, sizeof(int) * gSize);
    cudaMalloc((void**)&d_estimates, sizeof(int) * gSize);

    cudaMemcpy((void*)d_adj_mat, (void*)adjMat, sizeof(int) * gSize * gSize, cudaMemcpyHostToDevice);
    cudaMemset((void*)d_unvisited, 0, sizeof(int) * gSize);
    cudaMemset((void*)d_frontier, 0, sizeof(int) * gSize);
    cudaMemset((void*)d_estimates, 0, sizeof(int) * gSize);

    int num_blocks = (gSize / block_size) + 1;

    int* minimums = (int*)malloc(sizeof(int) * gSize);
    int* d_minimums;
    cudaMalloc((void**)&d_minimums, sizeof(int) * gSize);
    cudaMemset((void*)d_minimums, 0, sizeof(int) * gSize);

    float duration_par = 0;
    cudaEvent_t start_pararell, stop_pararell;
    cudaEventCreate(&start_pararell);
    cudaEventCreate(&stop_pararell);
    cudaEventRecord(start_pararell, 0);

    int min_i = INT_MAX;
    int* d_min_i;
    cudaMalloc((void**)&d_min_i, sizeof(int));

    init << <num_blocks, block_size >> > (d_unvisited, d_frontier, d_estimates, gSize);
    do {
        min_i = INT_MAX;
        relax_f << <num_blocks, block_size >> > (d_estimates, d_frontier, d_unvisited, d_adj_mat, gSize);
        find_min << <num_blocks, block_size, block_size * sizeof(int) >> > (d_unvisited, d_estimates, d_minimums, gSize);
        cudaMemcpy(minimums, d_minimums, gSize * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < num_blocks; i++)
            min_i = std::min(minimums[i], min_i);
        cudaMemcpy(d_min_i, &min_i, sizeof(int), cudaMemcpyHostToDevice);
        update << <num_blocks, block_size >> > (d_unvisited, d_frontier, d_estimates, d_min_i, gSize);
    } while (min_i != INT_MAX);

    cudaEventRecord(stop_pararell, 0);
    cudaEventSynchronize(stop_pararell);
    cudaEventElapsedTime(&duration_par, start_pararell, stop_pararell);
    cudaEventDestroy(start_pararell);
    cudaEventDestroy(stop_pararell);

    std::cout
        << "Parallel time: "
        << '\n'
        << duration_par
        << " ms "
        << std::endl;

    // copy results
    cudaMemcpy(shortest_out, d_estimates, sizeof(int) * gSize, cudaMemcpyDeviceToHost);

    auto start_seq = std::chrono::high_resolution_clock::now();
    long long duration_seq = 0;
    std::vector<NodeState> seq_distances_result{};
    seq_distances_result = dijkstra_seq(adj_matrix);
    duration_seq = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_seq).count();
    std::cout
        << "Dijkstra sequential time: "
        << '\n'
        << duration_seq
        << " ms "
        << std::endl;
	bool match = true;
	for (int i = 0; i < seq_distances_result.size(); i++) {
		if (seq_distances_result[i].distance != shortest_out[i]) {
			match = false;
		}
	}
	if (!match) std::cout << "Wrong" << std::endl;
	else std::cout << "Correct" << std::endl;

    output_file << std::to_string((int)duration_par) << ";" << std::to_string((int)duration_seq) << std::endl;

    cudaFree(d_minimums);
    cudaFree(d_adj_mat);
    cudaFree(d_unvisited);
    cudaFree(d_frontier);
    cudaFree(d_estimates);
    cudaFree(d_min_i);
    cudaFree(d_minimums);
    free(adjMat);
    free(shortest_out);
    free(minimums);
}

void run_bellman_ford(size_t size, std::ofstream& output_file, int block_size) {
    std::cout << '\n' << "Size: " << size << std::endl;

    // Prepare CUDA layout
    dim3 block_structure(block_size, 1, 1);
    dim3 grid_structure((size + block_size - 1) / block_size, 1, 1); // jest jeszcze opcja size / threads_per_block

    // Prepare host memory
    int* h_Mat = new int[size * size];

    // Generate adjency matrix - 2D vector
    std::vector<std::vector<int>> adj_matrix = generate_adj_matrix(size);
    std::vector<int> seq_distances_result{};

    // Parallel computations use 1D array, so copy 2D matrix
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            h_Mat[i * size + j] = adj_matrix[i][j];
        }
    }

    // Distances
    int* h_Dist = new int[size];
    for (int i = 0; i < size; i++) {
        h_Dist[i] = INT_MAX;
    }
    h_Dist[0] = 0;

    /////////////////////////////////// Problem z duration_seq, nie jest widoczne ponizej
    // Solve sequentially ford
    auto start_seq = std::chrono::high_resolution_clock::now();
    long long duration_seq = 0;
    seq_distances_result = bellman_ford_seq(adj_matrix);
    duration_seq = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_seq).count();
    std::cout
        << "Bellman-ford sequential time: "
        << '\n'
        << duration_seq
        << " ms "
        << std::endl;

    float duration_par = 0;
    // device memory
    int* d_Mat, * d_Dist;
    cudaMalloc((void**)&d_Mat, size * size * sizeof(int));
    cudaMalloc((void**)&d_Dist, size * sizeof(int));

    // load device memory from host memory
    cudaMemcpy(d_Mat, h_Mat, size * size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Dist, h_Dist, size * sizeof(int), cudaMemcpyHostToDevice);

    // run pararell algorithm
    cudaEvent_t start_pararell, stop_pararell;
    cudaEventCreate(&start_pararell);
    cudaEventCreate(&stop_pararell);
    cudaEventRecord(start_pararell, 0);

    for (int vertex = 0; vertex < size - 1; vertex++) {
        bellman_ford_parallel << <grid_structure, block_structure >> > (d_Mat, d_Dist, size);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop_pararell, 0);
    cudaEventSynchronize(stop_pararell);
    cudaEventElapsedTime(&duration_par, start_pararell, stop_pararell);
    cudaEventDestroy(start_pararell);
    cudaEventDestroy(stop_pararell);

    std::cout
        << "Parallel time: "
        << '\n'
        << duration_par
        << " ms "
        << std::endl;

    // get results
    cudaMemcpy(h_Mat, d_Mat, size * size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Dist, d_Dist, size * sizeof(int), cudaMemcpyDeviceToHost);

    bool match = true;
    for (int i = 0; i < seq_distances_result.size(); i++) {
        if (seq_distances_result[i] != h_Dist[i]) {
            match = false;
        }
    }
    if (!match) std::cout << "Wrong" << std::endl;
    else std::cout << "Correct" << std::endl;

    cudaFree(d_Mat);
    cudaFree(d_Dist);

    // save to csv file
    output_file << std::to_string((int)duration_par) << ";" << std::to_string((int)duration_seq) << std::endl;

    delete[] h_Mat;
    delete[] h_Dist;
}
