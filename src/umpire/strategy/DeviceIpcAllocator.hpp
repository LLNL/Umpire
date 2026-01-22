//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_DeviceIpcAllocator_HPP
#define UMPIRE_DeviceIpcAllocator_HPP

#include <atomic>
#include <string>
#include <unordered_map>

#include "umpire/Allocator.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/MPI.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/MemoryResourceTraits.hpp"
#include "umpire/util/Platform.hpp"

#if defined(UMPIRE_ENABLE_CUDA)
#include <cuda_runtime_api.h>
typedef cudaIpcMemHandle_t gpuIpcMemHandle_t;
#elif defined(UMPIRE_ENABLE_HIP)
#include <hip/hip_runtime_api.h>
typedef hipIpcMemHandle_t gpuIpcMemHandle_t;
#endif

namespace umpire {

namespace resource {
class HostSharedMemoryResource;
}

namespace strategy {

class DeviceIpcAllocator : public AllocationStrategy {
 public:
  DeviceIpcAllocator(const std::string& name, int id) noexcept;

  DeviceIpcAllocator(const std::string& name, int id, Allocator device_allocator,
                     MemoryResourceTraits::shared_scope scope, std::size_t shared_memory_size = 1024 * 1024) noexcept;

  ~DeviceIpcAllocator();

  void* allocate(std::size_t bytes) override;
  void deallocate(void* ptr, std::size_t size = 0) override;

  Platform getPlatform() noexcept override;

  MemoryResourceTraits getTraits() const noexcept override;

  MPI_Comm get_scope_communicator();

 private:
  strategy::AllocationStrategy* m_device_allocator;
  strategy::AllocationStrategy* m_shared_allocator;
  umpire::resource::HostSharedMemoryResource* m_shared_memory_resource;

  struct IpcHandleInfo {
    gpuIpcMemHandle_t handle;
    std::size_t size;
    int device_id;
    std::atomic<bool> is_initialized;
  };

  std::unordered_map<void*, std::string> m_allocation_names;

  int m_scope_rank;
  MPI_Comm m_scope_comm;
  bool m_is_scope_leader;
  int m_scope_color;

  void setup_shared_scope(MemoryResourceTraits::shared_scope scope);
  std::string generate_allocation_name(std::size_t size_in_bytes);
  void* create(const std::string& name, std::size_t size_in_bytes);
  void* import(const std::string& name);
  IpcHandleInfo* get_handle_info(const std::string& name);
  IpcHandleInfo* create_handle_info(const std::string& name, std::size_t size_in_bytes);
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_DeviceIpcAllocator_HPP
