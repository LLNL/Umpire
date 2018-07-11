//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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
#ifndef UMPIRE_AllocationRecord_HPP
#define UMPIRE_AllocationRecord_HPP

#include <cstddef>

#include <memory>

#include "umpire/tpl/simpool/FixedSizePool.hpp"
#include "umpire/tpl/simpool/StdAllocator.hpp"

namespace umpire {

namespace strategy {
  class AllocationStrategy;
}

namespace util {

struct AllocationRecord
{
  void* m_ptr;
  size_t m_size;
  std::shared_ptr<strategy::AllocationStrategy> m_strategy;
};

namespace {
  static FixedSizePool<AllocationRecord, StdAllocator> pool;
}

template<typename... Args>
inline
AllocationRecord*
makeAllocationRecord(Args... args)
{
  auto record = pool.allocate();

  return new (record) AllocationRecord{args...};
}

inline
void
deleteAllocationRecord(AllocationRecord* ptr)
{
  pool.deallocate(ptr);
}

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_AllocationRecord_HPP
