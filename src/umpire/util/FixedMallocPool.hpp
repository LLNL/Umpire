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
    FixedMallocPool(const size_t object_bytes,
                    const size_t objects_per_pool = 64 * sizeof(int) * 8);

    ~FixedMallocPool();

    void* allocate(size_t bytes = 0);
    void deallocate(void* ptr);

    size_t numPools() const noexcept;

  private:
    struct Pool {
      char* data;
      int* avail;
      size_t num_avail;
      Pool(const size_t object_bytes, const size_t objects_per_pool,
           const size_t avail_bytes);
    };

    void newPool();
    void* allocInPool(Pool& p) noexcept;

    size_t m_obj_bytes;
    size_t m_obj_per_pool;
    size_t m_data_bytes;
    size_t m_avail_length;
    size_t m_current_bytes;
    size_t m_highwatermark;
    std::vector<Pool> m_pool;
    // NOTE: struct Pool lacks a non-trivial destructor. If m_pool is
    // ever reduced in size, then .data and .avail have to be manually
    // deallocated to avoid a memory leak.
};

} // end namespace strategy
} // end namespace umpire

#endif // UMPIRE_FixedPool_HPP
