#pragma once
#include <vector>

struct NodeState {
    int distance;
    bool visited;
};

std::vector<std::vector<int>> generate_adj_matrix(size_t size);
std::vector<int> bellman_ford_seq(const std::vector<std::vector<int>>& matrix);
std::vector<NodeState> dijkstra_seq(const std::vector<std::vector<int>>& matrix);
