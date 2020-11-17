//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_Inspector_HPP
#define UMPIRE_Inspector_HPP

#include "umpire/util/AllocationRecord.hpp"

#include <memory>

namespace umpire {
namespace strategy {

class AllocationStrategy;

namespace mixins {

class Inspector
{
  public:
    Inspector();

    void registerAllocation(
        void* ptr,
        std::size_t size,
        Memory* strategy);

    // Deregisters the allocation if the strategy matches, otherwise throws an error
    util::AllocationRecord deregisterAllocation(
      void* ptr, Memory* strategy);

  protected:
    std::size_t m_current_size;
    std::size_t m_high_watermark;
    std::size_t m_allocation_count;
};

} // end of namespace mixins
} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_Inspector_HPP
