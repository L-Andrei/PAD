#include <iostream>
#include "matrix.hpp"
#include <chrono>
#include <random>

// Codigo feito para o benchmark.

using namespace std;

int main() {
    size_t size = 4096;
    
    Matrix<double> m1(size, size);
    Matrix<double> m2(size, size);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);

    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            m1(i, j) = dist(rng);
            m2(i, j) = dist(rng);
        }
    }

    auto start = std::chrono::high_resolution_clock::now();

    Matrix<double> res_optimized = m1 * m2;

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    
    std::cout << duration.count() << std::endl;

    return 0;
}
