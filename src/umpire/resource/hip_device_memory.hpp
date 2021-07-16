//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include "umpire/resource/memory_resource.hpp"
#include "umpire/alloc/hip_malloc_allocator.hpp"

namespace umpire {
namespace resource {

  /*!
   * \brief Concrete MemoryResource object that uses the template _allocator to
   * allocate and deallocate memory.
   */
template<bool Tracking=true>
class hip_device_memory :
  public memory_resource<umpire::resource::hip_platform>
{
  public: 

  template<int N=0>
  static hip_device_memory* get() {
    static hip_device_memory self{N};
    return &self;
  }

  void* allocate(std::size_t n) {
    int old_device;
    hipGetDevice(&old_device);
    hipSetDevice(device_);

    void* ret = umpire::alloc::hip_malloc_allocator::allocate(n);

    hipSetDevice(old_device);
    
    if constexpr(Tracking) {
      return this->track_allocation(this, ret, n);
    } else {
      return ret;
    }
  }

  void deallocate(void* ptr) {
    int old_device;
    hipGetDevice(&old_device);
    hipSetDevice(device_);

    if constexpr(Tracking) {
      this->untrack_allocation(ptr);
    }

    umpire::alloc::hip_malloc_allocator::deallocate(ptr);

    hipSetDevice(old_device);
  }

  camp::resources::Platform get_platform() override
  {
    return camp::resources::Platform::hip;
  }

  private:
    hip_device_memory(int device) :
      memory_resource<camp::resources::Platform::hip>{"DEVICE_" + std::to_string(device)},
      device_{device} {}

    ~hip_device_memory() = default;
    hip_device_memory(const hip_device_memory&) = delete;
    hip_device_memory& operator=(const hip_device_memory&) = delete;

    int device_;
};

} // end of namespace resource
} // end of namespace umpire