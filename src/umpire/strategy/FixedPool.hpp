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
#ifndef UMPIRE_FixedPool_HPP
#define UMPIRE_FixedPool_HPP

#include "umpire/alloc/MallocAllocator.hpp"

#include "umpire/strategy/AllocationStrategy.hpp"

#include <cstddef>

namespace umpire {
namespace strategy {

/*!
 * \brief Pool for fixed size allocations
 *
 * This AllocationStrategy provides an efficient pool for fixed size
 * allocations, and used to quickly allocate and deallocate objects.
 */
class FixedPool : public AllocationStrategy
{
public:
  FixedPool(const std::string& name, int id,
            Allocator allocator, const size_t object_size,
            const size_t objects_per_pool = 1024 * sizeof(int) * sizeof(char));

  void* allocate(size_t bytes) override final;
  void deallocate(void* ptr) override final;

  long getCurrentSize() const noexcept override final;
  long getHighWatermark() const noexcept override final;
  long getActualSize() const noexcept override final;
  Platform getPlatform() noexcept override final;

private:
  constexpr size_t bits_per_int = sizeof(int) * sizeof(char);

  struct Pool {
    char data[];
    int avail[];
    size_t num_avail;
    Pool(const size_t object_size, const size_t num_objects, FixedPool* fp);
  };

  size_t m_avail_bytes;
  size_t m_obj_bytes;
  size_t m_num_obj;
  AllocationStrategy* m_allocator;
  std::vector<Pool> m_pool;
};

} // end namespace strategy
} // end namespace umpire

#endif // UMPIRE_FixedPool_HPP
