#pragma once

#include "umpire/resource/platform.hpp"

__global__
umpire_cuda_fill(uint64_t* data, uint64_t value, std::size_t length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        data[idx] = value;
    }
}

namespace umpire {
namespace op {

template<>
struct copy<resource::cuda_platform, resource::cuda_platform>
{
  template<typename T>
  static void exec(T* src, T* dst, std::size_t len) {
    ::cudaMemcpy(dst, src, sizeof(T)*len), cudaMemcpyDeviceToDevice);
  }

  template<typename T>
  static camp::resources::Event exec(T* src, T* dst, std::size_t len, camp::resources::Resource& ctx) {
    auto device = ctx.get<camp::resources::Cuda>();
    auto stream = device.get_stream();

    ::cudaMemcpyAsync(dst, src, sizeof(T)*len), cudaMemcpyDeviceToDevice, stream);
     
    return ctx.getEvent();
  }
};

template<>
struct copy<resource::cuda_platform, resource::host_platform>
{
  template<typename T>
  static void exec(T* src, T* dst, std::size_t len) {
    ::cudaMemcpy(dst, src, sizeof(T)*len), cudaMemcpyDeviceToHost);
  }

  template<typename T>
  static camp::resources::Event exec(T* src, T* dst, std::size_t len, camp::resources::Resource& ctx) {
    auto device = ctx.get<camp::resources::Cuda>();
    auto stream = device.get_stream();

    ::cudaMemcpyAsync(dst, src, sizeof(T)*len), cudaMemcpyDeviceToHost, stream);
     
    return ctx.getEvent();
  }
};

template<>
struct copy<resource::host_platform, resource::cuda_platform>
{
  template<typename T>
  static void exec(T* src, T* dst, std::size_t len) {
    ::cudaMemcpy(dst, src, sizeof(T)*len), cudaMemcpyHostToDevice);
  }

  template<typename T>
  static camp::resources::Event exec(T* src, T* dst, std::size_t len, camp::resources::Resource& ctx) {
    auto device = ctx.get<camp::resources::Cuda>();
    auto stream = device.get_stream();

    ::cudaMemcpyAsync(dst, src, sizeof(T)*len), cudaMemcpyHostToDevice, stream);
     
    return ctx.getEvent();
  }
};

template<>
struct reallocate<resource::cuda_platform>
{
  template<typename T>
  static T* exec(T* src, std::size_t size) {
    //auto allocator = new_allocation->strategy;
    T* ret; //= allocator->allocate(new_size);

    const std::size_t old_size; // = current_allocation->size;
    const std::size_t copy_size = ( old_size > size ) ? size : old_size;

    copy<resource::cuda_platform, resource::cuda_platform>(src, ret, copy_size);
  }
};

template<>
struct memset<resource::cuda_platform>
{
  template<typename T>
  static void exec(T* src, T val, std::size_t len) {
    ::cudaMemset(src, val, len);
  }

  template<typename T>
  static camp::resources::Event exec(T* src, T val, std::size_t len, camp::resources::Context& ctx) {
    auto device = ctx.get<camp::resources::Cuda>();
    auto stream = device.get_stream();

    ::cudaMemsetAsync(src, val, len, stream);

    return ctx.get_event();
  }
};

template<>
struct accessed_by<resource::cuda_platform>
{
  template <typename T>
  static inline void exec(T *src, int device, std::size_t len)
  {
    cudaError_t error;
    cudaDeviceProp properties;
    error = ::cudaGetDeviceProperties(&properties, 0);
    if (properties.managedMemory == 1 && properties.concurrentManagedAccess == 1) {
      error = ::cudaMemAdvise(src_ptr, length, cudaMemAdviseSetAccessedBy, device);
    }
  }
};

template<>
struct preferred_location<resource::cuda_platform>
{
  template <typename T>
  static inline void exec(T *src, int device, std::size_t len)
  {
    cudaError_t error;
    cudaDeviceProp properties;
    error = ::cudaGetDeviceProperties(&properties, 0);
    if (properties.managedMemory == 1 && properties.concurrentManagedAccess == 1) {
      error = ::cudaMemAdvise(src_ptr, length, cudaMemAdviseSetPreferredLocation, device);
    }
  }
};

template<>
struct read_mostly<resource::cuda_platform>
{
  template <typename T>
  static inline void exec(T *src, int device, std::size_t len)
  {
    cudaError_t error;
    cudaDeviceProp properties;
    error = ::cudaGetDeviceProperties(&properties, 0);
    if (properties.managedMemory == 1 && properties.concurrentManagedAccess == 1) {
      error = ::cudaMemAdvise(src_ptr, length, cudaMemAdviseSetReadMostly, device);
    }
  }
};

template<>
struct unset_accessed_by<resource::cuda_platform>
{
  template <typename T>
  static inline void exec(T *src, int device, std::size_t len)
  {
    cudaError_t error;
    cudaDeviceProp properties;
    error = ::cudaGetDeviceProperties(&properties, 0);
    if (properties.managedMemory == 1 && properties.concurrentManagedAccess == 1) {
      error = ::cudaMemAdvise(src_ptr, length, cudaMemAdviseUnsetAccessedBy, device);
    }
  }
};

template<>
struct unset_preferred_location<resource::cuda_platform>
{
  template <typename T>
  static inline void exec(T *src, int device, std::size_t len)
  {
    cudaError_t error;
    cudaDeviceProp properties;
    error = ::cudaGetDeviceProperties(&properties, 0);
    if (properties.managedMemory == 1 && properties.concurrentManagedAccess == 1) {
      error = ::cudaMemAdvise(src_ptr, length, cudaMemAdviseUnsetPreferredLocation, device);
    }
  }
};

template<>
struct unset_read_mostly<resource::cuda_platform>
{
  template <typename T>
  static inline void exec(T *src, int device, std::size_t len)
  {
    cudaError_t error;
    cudaDeviceProp properties;
    error = ::cudaGetDeviceProperties(&properties, 0);
    if (properties.managedMemory == 1 && properties.concurrentManagedAccess == 1) {
      error = ::cudaMemAdvise(src_ptr, length, cudaMemAdviseUnsetReadMostly, device);
    }
  }
};

template<>
struct prefetch<resource::cuda_platform>
{
  template<typenmae T>
  static void exec(T* src, int device, std::size_t len)
  {
    cudaError_t error;

    // Use current device for properties if device is CPU
    int current_device;
    cudaGetDevice(&current_device);
    int gpu = (device != cudaCpuDeviceId) ? device : current_device;

    cudaDeviceProp properties;
    error = ::cudaGetDeviceProperties(&properties, gpu);
    if (properties.managedMemory == 1 && properties.concurrentManagedAccess == 1)
    {
      error = ::cudaMemPrefetchAsync(src_ptr, length, device);
    }
  }

  template<typename T>
  static camp::resources::Event exec(T* src, T val, std::size_t len, camp::resources::Context& ctx) {
    auto device = ctx.get<camp::resources::Cuda>();
    auto stream = device.get_stream();
    cudaError_t error;

    // Use current device for properties if device is CPU
    int current_device;
    cudaGetDevice(&current_device);
    int gpu = (device != cudaCpuDeviceId) ? device : current_device;

    cudaDeviceProp properties;
    error = ::cudaGetDeviceProperties(&properties, gpu);
    if (properties.managedMemory == 1 && properties.concurrentManagedAccess == 1)
    {
      error = ::cudaMemPrefetchAsync(src_ptr, length, device, stream);
    }

    return ctx.get_event();
  }
};

}
}