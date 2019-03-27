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

#include <memory>
#include <vector>

#include "umpire/strategy/AllocationStrategy.hpp"

#include "umpire/Allocator.hpp"

#include "umpire/tpl/simpool/StdAllocator.hpp"

namespace umpire {
namespace strategy {

/*!
 * \brief Pool for fixed size allocations
 *
 * This AllocationStrategy provides an efficient pool for fixed size
 * allocations of size T. Pools of NP objects of type T are constructed, and
 * used to quickly allocate and deallocate objects.
 */
template <typename T, int NP=64, typename IA=StdAllocator>
class FixedPool
  : public AllocationStrategy
{

  public:
    FixedPool(
        const std::string& name,
        int id,
        Allocator allocator);

    ~FixedPool();

    void finalize() override;

    void* allocate(size_t bytes) override;

    void deallocate(void* ptr) override;

    long getCurrentSize() const noexcept override;
    long getHighWatermark() const noexcept override;
    long getActualSize() const noexcept override;

    Platform getPlatform() noexcept override;

  private:
    struct Pool
    {
      unsigned char *data;
      unsigned int *avail;
      unsigned int numAvail;
      struct Pool* next;
    };

    void newPool(struct Pool **pnew);

    T* allocInPool(struct Pool *p);

    size_t numPools() const noexcept;

    void free();

    struct Pool *m_pool;
    size_t m_num_per_pool;
    size_t m_total_pool_size;

    size_t m_num_blocks;


    long m_highwatermark;
    long m_current_size;

    bool m_finalized;

    strategy::AllocationStrategy* m_allocator;
};

} // end of namespace strategy
} // end namespace umpire

#include "umpire/strategy/FixedPool.inl"

#endif // UMPIRE_FixedPool_HPP
