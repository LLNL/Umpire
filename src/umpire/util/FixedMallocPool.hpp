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
#ifndef UMPIRE_FixedMallocPool_HPP
#define UMPIRE_FixedMallocPool_HPP

#include "umpire/Allocator.hpp"

#include <cstdlib>
#include <vector>

namespace umpire {
namespace util {

/*!
 * \brief Pool for fixed size allocations using malloc()
 *
 * Another version of this class exists in umpire::strategy, but this
 * version does not rely on Allocator and all the memory tracking
 * statistics, so it is useful for building objects in umpire::util.
 */
class FixedMallocPool
{
public:
  FixedMallocPool(const std::size_t object_bytes,
                  const std::size_t objects_per_pool = 1024*1024);

  ~FixedMallocPool();

  void* allocate(std::size_t bytes = 0);
  void deallocate(void* ptr);

  std::size_t numPools() const noexcept;

private:
  struct Pool {
    unsigned char* data;
    unsigned char* next;
    unsigned int num_initialized;
    std::size_t num_free;
    Pool(const std::size_t object_bytes, const std::size_t objects_per_pool);
  };

  void newPool();
  void* allocInPool(Pool& p) noexcept;

  unsigned char* addr_from_index(const Pool& p, unsigned int i) const;
  unsigned int index_from_addr(const Pool& p, const unsigned char* ptr) const;

  const std::size_t m_obj_bytes;
  const std::size_t m_obj_per_pool;
  const std::size_t m_data_bytes;
  std::vector<Pool> m_pool;
  // NOTE: struct Pool lacks a non-trivial destructor. If m_pool is
  // ever reduced in size, then .data has to be manually deallocated
  // to avoid a memory leak.
};

} // end namespace strategy
} // end namespace umpire

#endif // UMPIRE_FixedPool_HPP
