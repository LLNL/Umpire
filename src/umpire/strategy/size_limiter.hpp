//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include "umpire/strategy/allocation_strategy.hpp"
#include "umpire/detail/log.hpp"

namespace umpire {
namespace strategy {

/*!
 *
 * \brief An allocator with a limited total size.
 *
 * Using this AllocationStrategy with another can be a good way to limit the
 * total size of allocations made on a particular resource or from a particular
 * context.
 */
template<typename Memory=memory, bool Tracking=true>
class size_limiter :
  public allocation_strategy
{
  public:
    size_limiter(
      const std::string& name,
      Memory* memory,
      std::size_t size_limit) :
      allocation_strategy(name),
      memory_{memory},
      size_limit_{size_limit},
      total_size_{0}
    {
      if (!Tracking) {
        UMPIRE_LOG(Warning, "size_limiter doesn't work correctly when Tracking=false.");
      }
    }

    void* allocate(std::size_t n) 
    {
      total_size_ += n;
      if (total_size_ > size_limit_) {
        total_size_ -= n;
        UMPIRE_ERROR("Size limit exceeded.");
      }

      void* p = memory_->allocate(n);
      if (Tracking) {
        return this->track_allocation(this, p, n);
      } else {
        return p;
      }
    }

    void deallocate(void* ptr)
    {
      if constexpr(Tracking) {
        total_size_ -= this->untrack_allocation(ptr).size;
      } 
      memory_->deallocate(ptr);
    }

    camp::resources::Platform get_platform()
    {
      return memory_->get_platform();
    }

  private:
    Memory* memory_;
    const std::size_t size_limit_;
    std::size_t total_size_;
};

} // end of namespace strategy
} // end namespace umpire
