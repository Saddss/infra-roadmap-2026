// W5 任务：vector add host 代码（含验证 + 计时）

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

extern __global__ void vector_add_kernel(const float* a, const float* b, float* c, int n);

int main() {
    const int N = 1 << 20;  // 1M elements
    const size_t bytes = N * sizeof(float);

    // TODO W5: malloc host 内存，初始化 a 和 b
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(2 * i);
    }

    // TODO W5: cudaMalloc 三个 device buffer
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    // TODO W5: cudaMemcpy host -> device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // TODO W5: 启动 kernel，block size 256，grid size 自动算
    int block = 256;
    int grid = (N + block - 1) / block;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    vector_add_kernel<<<grid, block>>>(d_a, d_b, d_c, N);

    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms = 0; cudaEventElapsedTime(&ms, start, stop);
    CHECK_CUDA(cudaGetLastError());

    // TODO W5: 拷回 + 验证
    CHECK_CUDA(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];
        if (abs(h_c[i] - expected) > 1e-5) errors++;
    }
    printf("Vector add %d elements: %s, %.3f ms, bandwidth %.1f GB/s\n",
           N, errors ? "FAIL" : "OK", ms, 3.0 * bytes / 1e9 / (ms / 1000));

    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return errors;
}
