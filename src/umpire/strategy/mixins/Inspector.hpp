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
        size_t size,
        strategy::AllocationStrategy* strategy);

    void deregisterAllocation(void* ptr);

  protected:
    long m_current_size{0};
    long m_high_watermark{0};
};

} // end of namespace mixins
} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_Inspector_HPP
