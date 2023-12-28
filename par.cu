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

// https://shreeraman-ak.medium.com/parallel-reduction-with-cuda-d0ae10c1ae2c
// https://link.springer.com/content/pdf/10.1007/978-3-642-01970-8_91.pdf

// Publikacja
// c - shortest distances
// F - nodes that can be processed at once cause the have the same min distance
// u - visited flags

//void initialize(c, f, u) {
//    forall vertex i{
//    c[i] = INFINITY;
//    f[i] = false;
//    u[i] = true;
//    }//for
//    c[0] = 0;
//    f[0] = true; u[0] = false;
//}

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

//void relax_F(c, f, u) {
//    forall i in parallel do {
//        if (f[i]) {
//            forall j successor of i do {
//                if (u[j])
//                    atomicMin(c[j], c[i] + w[i, j]);
//            }//for
//        }//if
//    }//for
//}

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

//void update(c, f, u, mssp) {
//    forall i in parallel do {
//        f[i] = false;
//        if (c[i] == mssp) {
//            u[i] = false;
//            f[i] = true;
//        }//if
//    }//for
//}

//__global__ void update(int* c, int* f, int* u, int* closest, size_t size) {
//    int id = blockIdx.x * blockDim.x + threadIdx.x;
//    if (id < size) {
//        f[id] = 0;
//        if (u[id] && c[id] == *closest) { // ?
//            u[id] = 0;
//            f[id] = 1;
//        }
//    }
//}

__global__ void update(int* U, int* F, int* d, int* del, size_t size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        F[id] = 0;
        if (U[id] && d[id] < del[0]) {
            U[id] = 0;
            F[id] = 1;
        }
    }
}

//void minimum1(u, c, minimums) {
//    forall i in parallel do {
//        thid = threadIdx.x;
//        i = blockIdx.x * (2 * blockDim.x) + threadIdx.x;
//        j = i + blockDim.x;
//        data1 = u[i] ? c[i] : INFINITY;
//        data2 = u[j] ? c[j] : INFINITY;
//        sdata[thid] = min(data1, data2);
//        __syncthreads();
//        for (s = blockDim.x / 2; s > 0; s >>= 1) {
//            if (thid < s) {
//                sdata[thid] = min(sdata[thid], sdata[thid + s]);
//            }// if
//            __syncthreads();
//        }// for
//        if (thid == 0) minimums[blockIdx.x] = sdata[0];
//    }// forall
//}

//__global__ void minimum1(int *u, int* c, int* minimums, int n) {
//    extern __shared__ int sdata[BLOCK_SIZE];
//    int thid = threadIdx.x;
//    int i = blockIdx.x * (2 * blockDim.x) + threadIdx.x;
//    int j = i + blockDim.x;
//    int data1 = (i < n) ? (u[i] ? c[i] : INT_MAX) : INT_MAX;
//    int data2 = (j < n) ? (u[j] ? c[j] : INT_MAX) : INT_MAX;
//    sdata[thid] = min(data1, data2);
//    __syncthreads();
//    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
//        if (thid < s) {
//            sdata[thid] = min(sdata[thid], sdata[thid + s]);
//        }
//        __syncthreads();
//    }
//    if (thid == 0) {
//       // printf("%d\n", sdata[0]);
//        minimums[blockIdx.x] = sdata[0];
//    }
//}

__global__ void min(int* U, int* d, int* outDel, int* minOutEdges, size_t gSize, int useD) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int pos1 = 2 * id;
    int pos2 = 2 * id + 1;
    int val1, val2;
    if (pos1 < gSize) {
        val1 = minOutEdges[pos1] + (useD ? d[pos1] : 0);
        if (pos2 < gSize) {
            val2 = minOutEdges[pos2] + (useD ? d[pos2] : 0);

            val1 = val1 <= 0 ? INT_MAX : val1;
            val2 = val2 <= 0 ? INT_MAX : val2;
            if (useD) {
                val1 = U[pos1] ? val1 : INT_MAX;
                val2 = U[pos2] ? val2 : INT_MAX;
            }
            if (val1 > val2) {
                outDel[id] = val2;
            }
            else {
                outDel[id] = val1;
            }
        }
        else {
            val1 = val1 <= 0 ? INT_MAX : val1;
            if (useD) {
                val1 = U[pos1] ? val1 : INT_MAX;
            }
            outDel[id] = val1;
        }
    }
}

__global__ void findAllMins(int* adjMat, int* outVec, size_t size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    //input matrix is one dimensional, so make sure to use correct offset
    int offset = id * size;
    int min = INT_MAX;
    if (id < size) {
        for (int i = 0; i < size; i++) {
            if (adjMat[offset + i] < min && adjMat[offset + i] > 0) {
                min = adjMat[offset + i];
            }
        }
        outVec[id] = min;
    }
}

void run_dijkstra(size_t size, std::ofstream& output_file, int block_size, bool run_seq = true, bool run_parallel = true) {
    size_t gSize = size;

    int* adjMat;
    int* shortestOut;

    int* d_adjMat;
    int* d_outVec;
    int* d_unvisited;
    int* d_frontier;
    int* d_estimates;
    int* d_delta;
    int* d_minOutEdge;

    std::vector<std::vector<int>> adj_matrix = generate_adj_matrix(size);
    adjMat = (int*)malloc(size * size * sizeof(int));
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            adjMat[i * size + j] = adj_matrix[i][j];
        }
    }

    shortestOut = (int*)malloc(sizeof(int) * gSize);

    cudaMalloc((void**)&d_adjMat, sizeof(int) * gSize * gSize);
    cudaMalloc((void**)&d_outVec, sizeof(int) * gSize);
    cudaMalloc((void**)&d_unvisited, sizeof(int) * gSize);
    cudaMalloc((void**)&d_frontier, sizeof(int) * gSize);
    cudaMalloc((void**)&d_estimates, sizeof(int) * gSize);
    cudaMalloc((void**)&d_minOutEdge, sizeof(int) * gSize);
    cudaMalloc((void**)&d_delta, sizeof(int) * gSize);

    cudaMemcpy((void*)d_adjMat, (void*)adjMat, sizeof(int) * gSize * gSize, cudaMemcpyHostToDevice);
    cudaMemset((void*)d_outVec, 0, sizeof(int) * gSize);
    cudaMemset((void*)d_unvisited, 0, sizeof(int) * gSize);
    cudaMemset((void*)d_frontier, 0, sizeof(int) * gSize);
    cudaMemset((void*)d_estimates, 0, sizeof(int) * gSize);
    cudaMemset((void*)d_minOutEdge, 0, sizeof(int) * gSize);


    int   del;
    int numBlocks = (gSize / block_size) + 1;

    findAllMins << <numBlocks, block_size >> > (d_adjMat, d_minOutEdge, gSize);

    //////////////////////////////////////////////////////////////////////
    int  curSize = gSize;
    int  dFlag;
    int* d_minTemp1;
    int* d_minTemp2;

    cudaMalloc((void**)&d_minTemp1, sizeof(int) * gSize);


    float duration_par = 0;
    cudaEvent_t start_pararell, stop_pararell;
    cudaEventCreate(&start_pararell);
    cudaEventCreate(&stop_pararell);
    cudaEventRecord(start_pararell, 0);

    init << <numBlocks, block_size >> > (d_unvisited, d_frontier, d_estimates, gSize);

    do {
        dFlag = 1;
        curSize = gSize;
        cudaMemcpy(d_minTemp1, d_minOutEdge, sizeof(int) * gSize, cudaMemcpyDeviceToDevice);

        // relax all frontiers
        relax_f << <numBlocks, block_size >> > (d_estimates, d_frontier, d_unvisited, d_adjMat, gSize);

        do {
            min << <numBlocks, block_size >> > (d_unvisited, d_estimates, d_delta, d_minTemp1, curSize, dFlag);
            d_minTemp2 = d_minTemp1;
            d_minTemp1 = d_delta;
            d_delta = d_minTemp2;
        
            curSize /= 2;
            dFlag = 0;
        } while (curSize > 0);
        
        d_minTemp2 = d_minTemp1;
        d_minTemp1 = d_delta;
        d_delta = d_minTemp2;

        //minimum1 << <numBlocks, block_size >> > (_d_unvisited, _d_estimates, _d_minTemp1, size);

        update << <numBlocks, block_size >> > (d_unvisited, d_frontier, d_estimates, d_delta, gSize);

        //cudaMemcpy(&del, _d_minOutEdge, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&del, d_delta, sizeof(int), cudaMemcpyDeviceToHost);
    } while (del != INT_MAX);

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
    cudaMemcpy(shortestOut, d_estimates, sizeof(int) * gSize, cudaMemcpyDeviceToHost);

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
		if (seq_distances_result[i].distance != shortestOut[i]) {
			match = false;
		}
	}
	if (!match) std::cout << "Wrong" << std::endl;
	else std::cout << "Correct" << std::endl;

    output_file << std::to_string((int)duration_par) << ";" << std::to_string((int)duration_seq) << std::endl;

    cudaFree(d_minTemp1);
    //////////////////////////////////////////////////////////////////////

    cudaFree(d_adjMat);
    cudaFree(d_outVec);
    cudaFree(d_unvisited);
    cudaFree(d_frontier);
    cudaFree(d_estimates);
    cudaFree(d_minOutEdge);
    cudaFree(d_delta);
    free(adjMat);
    free(shortestOut);
}

void run_bellman_ford(size_t size, std::ofstream& output_file, int block_size, bool run_seq = true, bool run_parallel = true) {
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
    if (run_seq) {
        seq_distances_result = bellman_ford_seq(adj_matrix);
        duration_seq = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_seq).count();
        std::cout
            << "Bellman-ford sequential time: "
            << '\n'
            << duration_seq
            << " ms "
            << std::endl;
    }

    float duration_par = 0;
    if (run_parallel) {
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

        if (run_seq) {
            bool match = true;
            for (int i = 0; i < seq_distances_result.size(); i++) {
                if (seq_distances_result[i] != h_Dist[i]) {
                    match = false;
                }
            }
            if (!match) std::cout << "Wrong" << std::endl;
            else std::cout << "Correct" << std::endl;
        }

        cudaFree(d_Mat);
        cudaFree(d_Dist);
    }

    // save to csv file
    output_file << std::to_string((int)duration_par) << ";" << std::to_string((int)duration_seq) << std::endl;

    delete[] h_Mat;
    delete[] h_Dist;
}
