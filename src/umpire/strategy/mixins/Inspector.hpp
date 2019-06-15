//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
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
        strategy::AllocationStrategy* strategy);

    // Deregisters the allocation if the strategy matches, otherwise throws an error
    util::AllocationRecord deregisterAllocation(
      void* ptr, strategy::AllocationStrategy* strategy);

  protected:
    std::size_t m_current_size;
    std::size_t m_high_watermark;
};

} // end of namespace mixins
} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_Inspector_HPP
