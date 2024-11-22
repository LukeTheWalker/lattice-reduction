#include "random_array.hpp"
#include "sequential_min.hpp"
#include "min_gpu.cuh"
#include <algorithm>
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono;

struct functions_to_test {
    void (*func)(const int*, int*, size_t);
    const char *name;
    size_t time_ms;
};

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <array size>" << endl;
        return 1;
    }

    vector<functions_to_test> functions = {
        {find_min_atomicMin_wrapper, "find_min_atomicMin", 0},
        {find_min_fixpoint_wrapper, "find_min_fixpoint", 0},
        {find_min_optimized_wrapper, "find_min_optimized", 0},
        {find_min_fixpoint_optimized_wrapper, "find_min_fixpoint_optimized", 0}
    };

    int seed = 42;
    size_t size = atol(argv[1]);

    printf("Array size: %ld\n", size);

    RandomArray ra(size, seed);

    auto start = high_resolution_clock::now();
    int min_algo = *min_element(ra.array, ra.array + size);
    auto end = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start).count();
    printf("Time taken for algorithm's implementation: %ld milliseconds, result %d\n", duration, min_algo);

    start = high_resolution_clock::now();
    int min_seq = find_min(ra.array, size);
    end = high_resolution_clock::now();
    if (min_algo != min_seq) cout << "Error: sequential implementation returned " << min_seq << " instead of " << min_algo << endl;
    printf("Time taken for sequential implementation: %ld milliseconds, result %d\n", duration_cast<milliseconds>(end - start).count(), min_seq);

    int *d_array = nullptr;
    start = high_resolution_clock::now();
    move_data_to_device((const int **) &ra.array, &d_array, size);
    end = high_resolution_clock::now();
    printf("Time taken to move data to device: %ld milliseconds\n", duration_cast<milliseconds>(end - start).count());

    for (auto &f : functions) {

        start = high_resolution_clock::now();
        int min_gpu = find_min_gpu(d_array, size, f.func);
        end = high_resolution_clock::now();

        f.time_ms = duration_cast<milliseconds>(end - start).count();

        if (min_algo != min_gpu) cout << "Error: " << f.name << " returned " << min_gpu << " instead of " << min_algo << endl;
        printf("Time taken for %s: %ld milliseconds, result %d\n", f.name, f.time_ms, min_gpu);
    }

    return 0;
}

