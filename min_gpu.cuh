#include <cuda_runtime.h>
#include <iostream>

void cuda_err_check(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

// Warp reduce using shuffle operations
__device__ __forceinline__ int warpReduceMin(int val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = min(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block reduce using shared memory and warp reduce
__device__ __forceinline__ int blockReduceMin(int val) {
    __shared__ int shared[32]; // Shared mem for 32 partial results
    
    size_t lane = threadIdx.x % 32;
    size_t wid = threadIdx.x / 32;
    
    // First warp reduction
    val = warpReduceMin(val);
    
    // Write reduced value to shared memory
    if (lane == 0) shared[wid] = val;
    
    __syncthreads();
    
    // Read from shared memory only if thread is in first warp
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : INT_MAX;
    
    // Final reduce within first warp
    if (wid == 0) {
        val = warpReduceMin(val);
    }
    
    return val;
}

__global__ void findMinKernel(const int* __restrict__ input, 
                            int* __restrict__ output,
                            const size_t N) {
    // Grid-stride loop for handling large arrays
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    size_t gridSize = blockDim.x * gridDim.x;
    
    // Local minimum for this thread
    int localMin = INT_MAX;
    
    // Grid stride loop
    for (size_t i = tid; i < N; i += gridSize) {
        localMin = min(localMin, input[i]);
    }
    
    // Reduce within block
    int blockMin = blockReduceMin(localMin);
    
    // Write result for this block to global memory
    if (threadIdx.x == 0) {
        atomicMin(output, blockMin);
    }
}

// Host function to launch the kernel
void find_min_optimized_wrapper(const int* input, int* output, size_t N) {
    // Set initial output value
    int initVal = INT_MAX;
    cudaMemcpy(output, &initVal, sizeof(int), cudaMemcpyHostToDevice);
    
    // Calculate grid dimensions
    size_t blockSize = 256;  // Use multiple of 32 for efficient warp operations
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    size_t numBlocks;

    if (32 * numSMs < (N + blockSize - 1) / blockSize) numBlocks = 32 * numSMs;
    else                                               numBlocks = (N + blockSize - 1) / blockSize;
    
    // Launch kernel
    findMinKernel<<<numBlocks, blockSize>>>(input, output, N);
}

__global__ void find_min_atomicMin_kernel(const int *d_array, int *d_min, size_t size)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < size)
    {
        atomicMin(d_min, d_array[i]);
    }
}

void find_min_atomicMin_wrapper(const int *d_array, int *d_min, size_t size)
{
    int blockSize = 1024;
    int gridSize = (size + blockSize - 1) / blockSize;
    find_min_atomicMin_kernel<<<gridSize, blockSize>>>(d_array, d_min, size);
}

__global__ void find_min_fixpoint_kernel(const int *d_array, int *d_min, size_t size)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size && d_array[i] < *d_min)
        *d_min = d_array[i];
}

void find_min_fixpoint_wrapper(const int *d_array, int *d_min, size_t size)
{
    cudaError_t err;
    int blockSize = 1024;
    int gridSize = (size + blockSize - 1) / blockSize;

    int h_min_prev, h_min;

    err = cudaMemcpy(&h_min, d_min, sizeof(int), cudaMemcpyDeviceToHost);
    cuda_err_check(err, __FILE__, __LINE__);

    int cnt = 0;

    do
    {
        h_min_prev = h_min;
        find_min_fixpoint_kernel<<<gridSize, blockSize>>>(d_array, d_min, size);
        err = cudaDeviceSynchronize();
        cuda_err_check(err, __FILE__, __LINE__);
        err = cudaGetLastError();
        cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemcpy(&h_min, d_min, sizeof(int), cudaMemcpyDeviceToHost);
        cuda_err_check(err, __FILE__, __LINE__);
        cnt++;
    } while (h_min != h_min_prev);

    // printf("Iterations: %d\n", cnt);
}
__global__ void find_min_strided_kernel(const int *d_array, int *d_min, size_t size)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;

    int localMin = INT_MAX;

    for (size_t idx = i; idx < size; idx += stride)
    {
        localMin = min(localMin, d_array[idx]);
    }

    atomicMin(d_min, localMin);
}

void find_min_strided_wrapper(const int *d_array, int *d_min, size_t size)
{
    int blockSize = 1024;
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    int gridSize = 32 * numSMs;

    find_min_strided_kernel<<<gridSize, blockSize>>>(d_array, d_min, size);
}

int find_min_gpu(int *d_array, size_t size, void (*findMinWrapper)(const int *, int *, size_t))
{
    int *d_min;
    int h_min = std::numeric_limits<int>::max();

    cudaError_t err;

    err = cudaMalloc(&d_min, sizeof(int));
    cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_min, &h_min, sizeof(int), cudaMemcpyHostToDevice);
    cuda_err_check(err, __FILE__, __LINE__);

    // int blockSize = 1024;
    // int gridSize = (size + blockSize - 1) / blockSize;
    // findMinKernel<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_array, d_min, size);

    findMinWrapper(d_array, d_min, size);

    err = cudaDeviceSynchronize();
    cuda_err_check(err, __FILE__, __LINE__);
    err = cudaGetLastError();
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMemcpy(&h_min, d_min, sizeof(int), cudaMemcpyDeviceToHost);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaFree(d_min);
    cuda_err_check(err, __FILE__, __LINE__);

    return h_min;
}

void move_data_to_device(const int **h_array, int **d_array, size_t size)
{
    cudaError_t err;
    std::cout << "Allocating " << size * sizeof(int) / 1024 / 1024 / 1024 << " gygabytes on device" << std::endl;
    err = cudaMalloc(d_array, size * sizeof(int));
    cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(*d_array, *h_array, size * sizeof(int), cudaMemcpyHostToDevice);
    cuda_err_check(err, __FILE__, __LINE__);
}