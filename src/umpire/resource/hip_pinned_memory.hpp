//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include "umpire/resource/memory_resource.hpp"
#include "umpire/alloc/hip_pinned_allocator.hpp"

namespace umpire {
namespace resource {

  /*!
   * \brief Concrete MemoryResource object that uses the template _allocator to
   * allocate and deallocate memory.
   */
template<bool Tracking=true>
class hip_pinned_memory :
  public memory_resource<umpire::resources::hip_platform>
{
  public: 

  static hip_pinned_memory* get() {
    static hip_pinned_memory self;
    return &self;
  }

  void* allocate(std::size_t n) {
    void* ret{umpire::alloc::hip_pinned_allocator::allocate(n)};
    if constexpr(Tracking) {
      return this->track_allocation(this, ret, n);
    } else {
      return ret;
    }
  }

  void deallocate(void* ptr) {
    if constexpr(Tracking) {
      this->untrack_allocation(ptr);
    }
    umpire::alloc::hip_pinned_allocator::deallocate(ptr);
  }

  camp::resources::Platform get_platform() override
  {
    return camp::resources::Platform::hip;
  }

  private:
    hip_pinned_memory() :
      memory_resource<umpire::resources::hip_platform>{"PINNED"}
    {}

    ~hip_pinned_memory() = default;
    hip_pinned_memory(const hip_pinned_memory&) = delete;
    hip_pinned_memory& operator=(const hip_pinned_memory&) = delete;
};

} // end of namespace resource
} // end of namespace umpire