//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include "umpire/resource/memory_resource.hpp"
#include "umpire/alloc/cuda_malloc_allocator.hpp"

namespace umpire {
namespace resource {

  /*!
   * \brief Concrete MemoryResource object that uses the template _allocator to
   * allocate and deallocate memory.
   */
template<bool Tracking=true>
class cuda_device_memory :
  public memory_resource<umpire::resources:cuda_platform>
{
  public: 

  template<int N=0>
  static cuda_device_memory* get() {
    static cuda_device_memory self{N};
    return &self;
  }

  void* allocate(std::size_t n) {
    int old_device;
    cudaGetDevice(&old_device);
    cudaSetDevice(device_);

    void* ret = umpire::alloc::cuda_malloc_allocator::allocate(n);

    cudaSetDevice(old_device);

    if constexpr(Tracking) {
      return this->track_allocation(this, ret, n);
    } else {
      return ret;
    }
  }

  void deallocate(void* ptr) {
    int old_device;
    cudaGetDevice(&old_device);
    cudaSetDevice(device_);

    if constexpr(Tracking) {
      this->untrack_allocation(ptr);
    }

    umpire::alloc::cuda_malloc_allocator::deallocate(ptr);

    cudaSetDevice(old_device);
  }

  camp::resources::Platform get_platform() override
  {
    return camp::resources::Platform::cuda;
  }

  private:
    cuda_device_memory(int device) :
      memory_resource<camp::resources::Platform::cuda>{"DEVICE_" + std::to_string(device)},
      device_{device} {}

    ~cuda_device_memory() = default;
    cuda_device_memory(const cuda_device_memory&) = delete;
    cuda_device_memory& operator=(const cuda_device_memory&) = delete;

    int device_;
};

} // end of namespace resource
} // end of namespace umpire