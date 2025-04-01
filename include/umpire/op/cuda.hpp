#pragma once

#include "umpire/op/detail/utils.hpp"
#include "umpire/resource/platform.hpp"
#include "umpire/util/Platform.hpp"
#include "umpire/util/error.hpp"

// Forward declaration of kernel for launching directly in device code if needed
extern "C" {
__global__ void umpire_cuda_fill(void* data, int value, std::size_t length);
}

namespace {
// Copy direction mapping via template specialization
template <typename SRC, typename DST>
struct get_kind;

template <>
struct get_kind<resource::cuda_platform, resource::host_platform> {
  static constexpr cudaMemcpyKind value = cudaMemcpyDeviceToHost;
};

template <>
struct get_kind<resource::host_platform, resource::cuda_platform> {
  static constexpr cudaMemcpyKind value = cudaMemcpyHostToDevice;
};

template <>
struct get_kind<resource::cuda_platform, resource::cuda_platform> {
  static constexpr cudaMemcpyKind value = cudaMemcpyDeviceToDevice;
};
} // namespace

namespace umpire {
namespace op {

// CUDA implementation helpers
namespace {
// Helper function to check if a CUDA device supports managed memory features
inline bool check_device_managed_memory(int device)
{
  cudaDeviceProp properties;
  cudaError_t error = ::cudaGetDeviceProperties(&properties, device);

  if (error != cudaSuccess) {
    UMPIRE_ERROR(runtime_error, umpire::fmt::format("cudaGetDeviceProperties for device {} failed with error: {}",
                                                    device, cudaGetErrorString(error)));
  }

  return (properties.managedMemory == 1 && properties.concurrentManagedAccess == 1);
}

// Memory advice operation helper
template <typename T>
inline void advise_impl(T* ptr, std::size_t count, int device, cudaMemoryAdvise advice)
{
  if (!check_device_managed_memory(device))
    return;

  std::size_t size = detail::calculate_size(ptr, count);
  cudaError_t error = ::cudaMemAdvise(ptr, size, advice, device);

  if (error != cudaSuccess) {
    UMPIRE_ERROR(runtime_error,
                 umpire::fmt::format("cudaMemAdvise(ptr={}, size={}, advice={}, device={}) failed with error: {}", ptr,
                                     size, static_cast<int>(advice), device, cudaGetErrorString(error)));
  }
}

// CUDA copy implementation
template <typename T>
inline void copy_impl(T* src, T* dst, std::size_t count, cudaMemcpyKind kind)
{
  std::size_t size = detail::get_size<T>(count);

  cudaError_t error = ::cudaMemcpy(dst, src, size, kind);
  if (error != cudaSuccess) {
    UMPIRE_ERROR(runtime_error,
                 umpire::fmt::format("cudaMemcpy(dst={}, src={}, size={}, kind={}) failed with error: {}", dst, src,
                                     size, static_cast<int>(kind), cudaGetErrorString(error)));
  }
}

// CUDA async copy implementation
template <typename T>
inline camp::resources::EventProxy<camp::resources::Resource> copy_async_impl(T* src, T* dst, std::size_t count,
                                                                              camp::resources::Resource& r,
                                                                              cudaMemcpyKind kind)
{
  auto device = r.try_get<camp::resources::Cuda>();
  if (!device) {
    UMPIRE_ERROR(resource_error, umpire::fmt::format("Expected resources::Cuda, got resources::{}",
                                                     platform_to_string(r.get_platform())));
  }
  auto stream = device->get_stream();
  std::size_t size = detail::get_size<T>(count);

  cudaError_t error = ::cudaMemcpyAsync(dst, src, size, kind, stream);
  if (error != cudaSuccess) {
    UMPIRE_ERROR(
        runtime_error,
        umpire::fmt::format("cudaMemcpyAsync(dst={}, src={}, size={}, kind={}, stream={}) failed with error: {}", dst,
                            src, size, static_cast<int>(kind), (void*)stream, cudaGetErrorString(error)));
  }

  return camp::resources::EventProxy<camp::resources::Resource>{r};
}

// CUDA memset implementation
template <typename T>
inline void memset_impl(T* ptr, int value, std::size_t count)
{
  std::size_t size = detail::get_size<T>(count);

  cudaError_t error = ::cudaMemset(ptr, value, size);
  if (error != cudaSuccess) {
    UMPIRE_ERROR(runtime_error, umpire::fmt::format("cudaMemset(ptr={}, value={}, size={}) failed with error: {}", ptr,
                                                    value, size, cudaGetErrorString(error)));
  }
}

// CUDA async memset implementation
template <typename T>
inline camp::resources::EventProxy<camp::resources::Resource> memset_async_impl(T* ptr, int value, std::size_t count,
                                                                                camp::resources::Resource& r)
{
  auto device = r.try_get<camp::resources::Cuda>();
  if (!device) {
    UMPIRE_ERROR(resource_error, umpire::fmt::format("Expected resources::Cuda, got resources::{}",
                                                     platform_to_string(r.get_platform())));
  }
  auto stream = device->get_stream();
  std::size_t size = detail::get_size<T>(count);

  cudaError_t error = ::cudaMemsetAsync(ptr, value, size, stream);
  if (error != cudaSuccess) {
    UMPIRE_ERROR(runtime_error,
                 umpire::fmt::format("cudaMemsetAsync(ptr={}, value={}, size={}, stream={}) failed with error: {}", ptr,
                                     value, size, (void*)stream, cudaGetErrorString(error)));
  }

  return camp::resources::EventProxy<camp::resources::Resource>{r};
}

// Prefetch implementation
template <typename T>
inline void prefetch_impl(T* ptr, int device, std::size_t count)
{
  // Use current device for properties if device is CPU
  int current_device;
  cudaGetDevice(&current_device);
  int gpu = (device != cudaCpuDeviceId) ? device : current_device;

  if (check_device_managed_memory(gpu)) {
    std::size_t size = detail::get_size<T>(count);
    cudaError_t error = ::cudaMemPrefetchAsync(ptr, size, device, nullptr);

    if (error != cudaSuccess) {
      UMPIRE_ERROR(runtime_error,
                   umpire::fmt::format("cudaMemPrefetchAsync(ptr={}, size={}, device={}) failed with error: {}", ptr,
                                       size, device, cudaGetErrorString(error)));
    }
  }
}

// Async prefetch implementation
template <typename T>
inline camp::resources::EventProxy<camp::resources::Resource> prefetch_async_impl(T* ptr, int device, std::size_t count,
                                                                                  camp::resources::Resource& r)
{
  auto cuda_device = r.try_get<camp::resources::Cuda>();
  if (!cuda_device) {
    UMPIRE_ERROR(resource_error, umpire::fmt::format("Expected resources::Cuda, got resources::{}",
                                                     platform_to_string(r.get_platform())));
  }
  auto stream = cuda_device->get_stream();

  // Use current device for properties if device is CPU
  int current_device;
  cudaGetDevice(&current_device);
  int gpu = (device != cudaCpuDeviceId) ? device : current_device;

  if (check_device_managed_memory(gpu)) {
    std::size_t size = detail::get_size<T>(count);
    cudaError_t error = ::cudaMemPrefetchAsync(ptr, size, device, stream);

    if (error != cudaSuccess) {
      UMPIRE_ERROR(
          runtime_error,
          umpire::fmt::format("cudaMemPrefetchAsync(ptr={}, size={}, device={}, stream={}) failed with error: {}", ptr,
                              size, device, (void*)stream, cudaGetErrorString(error)));
    }
  }

  return camp::resources::EventProxy<camp::resources::Resource>{r};
}
