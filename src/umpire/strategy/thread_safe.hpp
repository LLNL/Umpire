//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include "umpire/strategy/allocation_strategy.hpp"

#include <mutex>

namespace umpire {
namespace strategy {

/*!
 *
 * \brief Make an Allocator thread safe
 *
 * Using this AllocationStrategy will make the provided allocator thread-safe
 * by syncronizing access to the allocators interface.
 */
template<typename Memory=allocation_strategy, 
         bool Tracking=true>
class thread_safe :
  public allocation_strategy
{
  public:
    using platform = typename Memory::platform;

    thread_safe(
        const std::string& name,
        Memory* memory) :
      allocation_strategy(name),
      memory_{memory},
      mutex_{}
    {
    }

    void* allocate(std::size_t n) final
    {
      std::lock_guard<std::mutex> lock(mutex_);
      void *p{memory_->allocate(n)};
      if constexpr(Tracking) {
        return this->track_allocation(this, p, n);
      } else {
        return p; 
      }
    }

    void deallocate(void* p) final
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if constexpr(Tracking) {
        this->untrack_allocation(p);
      }
      memory_->deallocate(p);
    }

    camp::resources::Platform get_platform() final
    {
      return memory_->get_platform();
    }

  protected:
    Memory* memory_;
    std::mutex mutex_;
};

} // end of namespace strategy
} // end of namespace umpire
