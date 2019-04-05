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
#ifndef UMPIRE_SizeLimiter_HPP
#define UMPIRE_SizeLimiter_HPP

#include <memory>

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/Allocator.hpp"

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
class SizeLimiter :
  public AllocationStrategy
{
  public:
      SizeLimiter(
        const std::string& name,
        int id,
        Allocator allocator,
        size_t size_limit);

    void* allocate(size_t bytes) override;
    void deallocate(void* ptr) override;

    long getCurrentSize() const noexcept override;
    long getHighWatermark() const noexcept override;

    Platform getPlatform() noexcept override;
  private:
    strategy::AllocationStrategy* m_allocator;

    size_t m_size_limit;
    size_t m_total_size;
};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_SizeLimiter_HPP
