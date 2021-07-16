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
class named :
  public allocation_strategy
{
  public:
    named(const std::string& name,
          Memory* memory) :
      allocation_strategy{name},
      memory_{memory}
    {
    }

    void* allocate(std::size_t n) override
    {
      void* p = memory_->allocate(n);
      if constexpr(Tracking) {
        return this->track_allocation(this, p, n);
      } else {
        return p;
      }
    }

    void deallocate(void* p) override
    {
      if constexpr(Tracking) {
        this->untrack_allocation(p);
      }
      memory_->deallocate(p);
    }

    camp::resources::Platform get_platform() 
    {
      return memory_->get_platform();
    }

  private:
    Memory* memory_;
};

} // end of namespace strategy
} // end of namespace umpire
