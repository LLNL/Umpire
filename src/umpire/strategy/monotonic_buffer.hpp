//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include "umpire/strategy/allocation_strategy.hpp"

namespace umpire {
namespace strategy {

template<typename Memory=allocation_strategy, bool Tracking=true>
class monotonic_buffer :
  public allocation_strategy
{
  public:
    using platform = typename Memory::platform;

    monotonic_buffer(
        const std::string& name,
        int id,
        Memory* memory,
        std::size_t capacity) :
      capacity_{capacity},
      block_{static_cast<char*>(memory_->allocate(capacity_))}
    {}

    ~monotonic_buffer()
    {
      memory_->deallocate(block_);
    }

    void* allocate(std::size_t n) override
    {
      void* ret{static_cast<void*>(block_ + current_size_)};
      current_size_ += n;

      // TODO: error if size > capacity

      last_size_ = n;

      if constexpr(Tracking) {
        return this->track_allocation(this, ret, n);
      } else {
        return ret;
      }
    }

    void deallocate(void* ptr) override 
    {
      if (ptr != (block_ + current_size_ - last_size_)) {
        // error
      }

      current_size_ -= last_size_;
      last_size_ = 0;

      if constexpr(Tracking) {
        this->untrack_allocation(ptr);
      }
    }

    camp::resources::Platform get_platform()
    {
      return memory_->get_platform();
    }

  private:
    std::size_t current_size_{0};
    std::size_t last_size_{0};
    const std::size_t capacity_{0};
    char* block_{nullptr};
    Memory* memory_{nullptr};
};

} // end of namespace strategy
} // end of namespace umpire
