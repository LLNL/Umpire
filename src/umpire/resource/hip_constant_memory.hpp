//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include "umpire/resource/memory_resource.hpp"
#include <hip/hip_runtime.h>

namespace umpire {
namespace resource {

template<bool Tracking=true>
class hip_constant_memory :
  public memory_resource<umpire::resources::hip_platform>
{
  public:

  static hip_constant_memory* get()
  {
    static hip_constant_memory self;
    return &self;
  }

  void* allocate(std::size_t)
  {
    is_allocated_ = true;
    if constexpr(Tracking) {
      return this->track_allocation(this, ptr_, 64*1024);
    } else {
      return ptr_;
    }
  }

  void deallocate(void*)
  {
    if (is_allocated_) {
      is_allocated_ = false;
    }
    if constexpr(Tracking) {
      this->untrack_allocation(ptr_);
    }
  }

  private:

  hip_constant_memory() :
    memory_resource<umpire::resource::hip>{"CONSTANT"}
    ptr_{s_umpire_internal_device_constant_memory}
  {}

  ~hip_constant_memory() = default;
  hip_constant_memory(const hip_constant_memory&) = delete;
  hip_constant_memory& operator=(const hip_constant_memory&) = delete;

  void* ptr_;
  std::once_flag m_initialized;
  bool is_allocated_{false};
};

} // end of namespace resource
} // end of namespace umpire
