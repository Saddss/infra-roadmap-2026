// W6 任务：实现 tiled matmul（PMPP 第 5 章）
//
// 核心思想：用 shared memory 把 A 和 B 的 tile 加载进来，每个 thread 算一格
// 收益：减少全局内存访问次数 ~2*BLOCK_SIZE 倍
//
// 提示：
//   - 用 __shared__ float As[TILE][TILE]; Bs[TILE][TILE];
//   - 外层循环：phase = 0..N/TILE
//   - 每个 phase 加载一个 tile，__syncthreads()，再算
//   - 注意边界（N 不能被 TILE 整除时）

#define TILE 16

__global__ void tiled_matmul_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    // TODO W6:
    //   1. 声明 __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    //   2. row = blockIdx.y * TILE + threadIdx.y
    //      col = blockIdx.x * TILE + threadIdx.x
    //   3. float sum = 0;
    //   4. for (int phase = 0; phase < (K+TILE-1)/TILE; phase++) {
    //        加载 As[ty][tx] = A[row, phase*TILE+tx]
    //        加载 Bs[ty][tx] = B[phase*TILE+ty, col]
    //        __syncthreads();
    //        for (int k = 0; k < TILE; k++) sum += As[ty][k] * Bs[k][tx];
    //        __syncthreads();
    //      }
    //   5. C[row, col] = sum
}

// 朴素版作为对比基线
__global__ void naive_matmul_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0;
    for (int k = 0; k < K; k++) sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}
