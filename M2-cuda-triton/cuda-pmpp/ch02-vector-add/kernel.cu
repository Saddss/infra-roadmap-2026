// W5 任务：实现 vector add CUDA kernel
// 参考 PMPP 第 2 章
//
// 提示：
//   - 每个 thread 负责一个元素
//   - 用 blockIdx.x * blockDim.x + threadIdx.x 计算全局索引
//   - 加边界判断：if (idx < n)

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    // TODO W5: 计算 idx，做边界检查，c[idx] = a[idx] + b[idx]
}
