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
#ifndef UMPIRE_DefaultAllocationStrategy_HPP
#define UMPIRE_DefaultAllocationStrategy_HPP

#include "umpire/strategy/AllocationStrategy.hpp"

namespace umpire {
namespace strategy {

class DefaultAllocationStrategy :
  public AllocationStrategy
{
  public:
    DefaultAllocationStrategy(std::shared_ptr<AllocationStrategy> allocator);

    void* allocate(size_t bytes);

    /*!
     * \brief Free the memory at ptr.
     *
     * \param ptr Pointer to free.
     */
    void deallocate(void* ptr);

    long getCurrentSize() const;
    long getHighWatermark() const;
    void coalesce() noexcept;

    Platform getPlatform();

  protected:
    std::shared_ptr<AllocationStrategy> m_allocator;
};

} // end of namespace strategy
} // end of namespace umpire

#endif
