#include <iostream>
#include <limits>
#include <algorithm>
#include "seq.h"

std::vector<std::vector<int>> generate_adj_matrix(size_t size) {
    srand(1);
    std::vector<std::vector<int>> matrix;
    matrix.reserve(size);
    for (int i = 0; i < size; i++) {
        std::vector<int> row(size, 0);
        for (int j = 0; j < size; j++)
            row[j] = static_cast<int>((rand() % 99) + 10);
        matrix.push_back(row);
    }
    return matrix;
}

std::vector<int> bellman_ford_seq(const std::vector<std::vector<int>>& matrix) {
    std::vector<int> distances(matrix.size(), INT_MAX);
    distances[0] = 0;
    for (int k = 0; k < matrix.size() - 1; k++) {
        for (int i = 0; i < matrix.size(); i++) {
            for (int j = 0; j < matrix.size(); j++) {
                if (matrix[i][j] != 0 && distances[i] != INT_MAX && distances[i] + matrix[i][j] < distances[j])
                    distances[j] = distances[i] + matrix[i][j];
            }
        }
    }
    return distances;
}

std::vector<NodeState> dijkstra_seq(const std::vector<std::vector<int>>& matrix) {
    const int starting_node = 0;
    std::vector<NodeState> node_state(matrix.size(), NodeState{ INT_MAX, false });
    node_state[starting_node].distance = 0;
    for (int i = 0; i < matrix.size(); i++) {
        int min_i = std::distance(node_state.begin(),
            std::min_element(node_state.begin(), node_state.end(), [](const NodeState& n1, const NodeState& n2) {
                if (!n1.visited && !n2.visited)
                    return n1.distance < n2.distance;
                else if (!n1.visited)
                    return true;
                else
                    return false;
                }
        ));
        node_state[min_i].visited = true;
        for (int neighbor = 0; neighbor < matrix.size(); neighbor++) {
            if (!node_state[neighbor].visited && matrix[min_i][neighbor] != 0 && node_state[min_i].distance != INT_MAX) {
                int distance = node_state[min_i].distance + matrix[min_i][neighbor];
                if (distance < node_state[neighbor].distance) {
                    node_state[neighbor].distance = node_state[min_i].distance + matrix[min_i][neighbor];
                }
            }
        }
    }
    return node_state;
}

