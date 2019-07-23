//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_DefaultAllocationStrategy_HPP
#define UMPIRE_DefaultAllocationStrategy_HPP

#include "umpire/strategy/AllocationStrategy.hpp"

namespace umpire {
namespace strategy {

class DefaultAllocationStrategy :
  public AllocationStrategy
{
  public:
    DefaultAllocationStrategy(strategy::AllocationStrategy* allocator);

    void* allocate(std::size_t bytes);

    /*!
     * \brief Free the memory at ptr.
     *
     * \param ptr Pointer to free.
     */
    void deallocate(void* ptr);

    long getCurrentSize() const;
    long getHighWatermark() const;

    Platform getPlatform();

  protected:
    strategy::AllocationStrategy* m_allocator;
};

} // end of namespace strategy
} // end of namespace umpire

#endif
