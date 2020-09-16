#pragma once

#include "umpire/config.hpp"
#include "umpire/util/MemoryResourceTraits.hpp"

namespace {

struct host_platform_tag {};

template <typename Platform>
struct accessor {};

template<>
struct accessor<host_platform_tag>
{
  template<typename T>
  void read(T* ptr)
  {
    volatile T ref;
    ref = *ptr;
    ref++;
  }

  template<typename T>
  void write(T* ptr, T val=0)
  {
    *ptr = val;
  }

  void reset() { }

  template<typename T>
  void verify(T* ptr, T val) {
    ASSERT_EQ(*ptr, val);
  }
};

#if defined(UMPIRE_ENABLE_CUDA) || defined(UMPIRE_ENABLE_HIP)

template <typename T>
__global__ void reader(T* ptr)
  {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx == 0) {
      volatile T ref;
      ref = *ptr;
      ref++;
    }
  }

template <typename T>
__global__ void writer(T* ptr, T val)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx == 0) {
    *ptr = val;
  }
}

template <typename T>
__global__ void verifier(T* ptr, T val)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx == 0) {
    if (*ptr != val) {
      asm("trap;");
    }
  }
}

#if defined(UMPIRE_ENABLE_CUDA)

struct cuda_platform_tag{};

template<>
struct accessor<cuda_platform_tag>
{

  template<typename T>
  void read(T* ptr)
  {
    reader<<<1, 16>>>(ptr);
    cudaDeviceSynchronize();
  }
  
  template<typename T>
  void write(T* ptr, T val = 0)
  {
    writer<<<1, 16>>>(ptr, val);
    cudaDeviceSynchronize();
  }

  template<typename T>
  void verify(T* ptr, T val) {
    verifier<<<1, 16>>>(ptr, val);
    cudaDeviceSynchronize();
  }

  void reset() {
    cudaDeviceReset();
  }
};
#endif

#if defined(UMPIRE_ENABLE_HIP)
template<>
struct accessor<hip_platform_tag>
{
  template<typename T>
  void read(T* ptr)
  {
    hipLaunchKernelGGL(reader, dim3(1), dim3(16), 0,0, ptr);
    hipDeviceSynchronize();
  }
  
  template<typename T>
  void write(T* ptr)
  {
    hipLaunchKernelGGL(writer, dim3(1), dim3(16), 0,0, ptr);
    hipDeviceSynchronize();
  }
};
#endif
#endif

}


namespace umpire {

using host = host_platform_tag;
#if defined(UMPIRE_ENABLE_CUDA)
using device = cuda_platform_tag;
#endif
#if defined(UMPIRE_ENABLE_HIP)
using device = hip_platform_tag;
#endif

void read(MemoryResourceTraits::resource_type r, double* data)
{
  if (r == MemoryResourceTraits::resource_type::HOST ) {
    accessor<host> a;
    a.read(data);
  } else if ((r == MemoryResourceTraits::resource_type::DEVICE) ||
            (r == MemoryResourceTraits::resource_type::PINNED) ||
            (r == MemoryResourceTraits::resource_type::UM)) {
    accessor<device> a;
    a.read(data);
  }
}

void write(MemoryResourceTraits::resource_type r, double* data)
{
  if (r == MemoryResourceTraits::resource_type::HOST ) {
    accessor<host> a;
    a.read(data);
  } else if ((r == MemoryResourceTraits::resource_type::DEVICE) ||
            (r == MemoryResourceTraits::resource_type::PINNED) ||
            (r == MemoryResourceTraits::resource_type::UM)) {
    accessor<device> a;
    a.read(data);
  }
}

}