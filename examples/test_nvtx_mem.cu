#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/QuickPool.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>

#define ASSERT_RT(Code) assert((Code) == cudaSuccess)

__global__ void InitArray(uint8_t* v, uint32_t size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= 0 && i < size)
    {
        v[i] = static_cast<uint8_t>(i);
    }
}

__global__ void TripleArray(uint8_t* v, uint32_t size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= 0 && i < size)
    {
        v[i] *= 3u;
    }
}

int main(void)
{

  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("DEVICE");
  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("NVTX_POOL", allocator);

    auto alloc = (uint8_t*) pool.allocate(63);
    InitArray<<<1, 64>>>(alloc, 63);
    ASSERT_RT(cudaDeviceSynchronize());

    // Violation: last byte out of bounds
    TripleArray<<<1, 64>>>(alloc, 64);
    ASSERT_RT(cudaDeviceSynchronize());

    // Violation: access after free
    pool.deallocate(alloc);
    TripleArray<<<1, 64>>>(alloc, 1);
    ASSERT_RT(cudaDeviceSynchronize());
}
