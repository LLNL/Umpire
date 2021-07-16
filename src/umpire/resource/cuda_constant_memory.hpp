//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include "umpire/resource/memory_resource.hpp"
#include <cuda_runtime_api.h>
#include <mutex>

namespace umpire {
namespace resource {

template <bool Tracking=true>
class cuda_constant_memory :
  public memory_resource<umpire::resource:cuda_platform>
{
  public:

  static cuda_constant_memory* get()
  {
    static cuda_constant_memory self;
    return &self;
  }

  void* allocate(std::size_t)
  {
    std::call_once(initialized_, [&] () {
      cudaError_t error = ::cudaGetSymbolAddress((void**)&ptr_, s_umpire_internal_device_constant_memory);
      if (error != cudaSuccess) {
        // UMPIRE_ERROR("cudaGetSymbolAddress failed with error: " << cudaGetErrorString(error));
        // error
      }
    });

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

  camp::resources::Platform get_platform() override
  {
    return camp::resources::Platform::cuda;
  }

  private:
  cuda_constant_memory() :
    public memory_resource<umpire::resource:cuda_platform>{"CONSTANT"}
  {}

  ~cuda_constant_memory() = default;
  cuda_constant_memory(const cuda_constant_memory&) = delete;
  cuda_constant_memory& operator=(const cuda_constant_memory&) = delete;

  void* ptr_;
  std::once_flag initialized_;
  bool is_allocated_;
};

} // end of namespace resource
} // end of namespace umpire
