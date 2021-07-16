//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include <memory>
#include <vector>

#include "umpire/strategy/allocation_strategy.hpp"
#include "umpire/detail/tracker.hpp"

namespace umpire {
namespace strategy {

template<typename Memory=memory, bool Tracking=true>
class slot_pool :
  public allocation_strategy
{
  public:
    using platform = typename Memory::platform;

    slot_pool(
      const std::string& name,
      Memory* memory,
      std::size_t slots) :
        allocation_strategy(name),
        slots_{slots},
        memory_{memory} 
    {
      sizes_ = new int64_t[slots_];
      pointers_ = new void*[slots_];

      for (std::size_t i = 0; i < slots_; ++i) {
        pointers_[i] = nullptr;
        sizes_[i] = 0;
      }
    }

    ~slot_pool() {
      for (std::size_t i = 0; i < slots_; ++i) {
        if (pointers_[i]) {
          memory_->deallocate(pointers_[i]);
        }
      }

      delete[] sizes_;
      delete[] pointers_;
    }

    void* allocate(std::size_t n) final
    {
      void *ptr{nullptr};
      int64_t int_n{static_cast<int64_t>(n)};

      for (std::size_t i = 0; i < slots_; ++i) {
        if (sizes_[i] == int_n) {
          sizes_[i] = -sizes_[i];
          ptr = pointers_[i];
          break;
        } else if (sizes_[i] == 0) {
          sizes_[i] = -int_n;
          pointers_[i] = memory_->allocate(n);
          ptr = pointers_[i];
          break;
        }
      }

      if constexpr(Tracking) {
        return this->track_allocation(this, ptr, n);
      } else {
        return ptr;
      }
    }

    void deallocate(void* ptr) final
    {
      if constexpr(Tracking) {
        this->untrack_allocation(ptr);
      }

      for (std::size_t i = 0; i < slots_; ++i) {
        if (pointers_[i] == ptr) {
          ptr = nullptr;
          sizes_[i] = -sizes_[i];
          break;
        }
      }
    }

    camp::resources::Platform get_platform() noexcept final
    {
      return memory_->get_platform();
    }

  private:
    void** pointers_;
    int64_t* sizes_;
    std::size_t slots_;
    Memory* memory_;
};

} // end of namespace strategy
} // end namespace umpire