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

#include "umpire/util/Macros.hpp"
#include "umpire/strategy/MixedPool.hpp"

namespace umpire {
namespace strategy {

namespace {

using PoolVector = std::vector< std::shared_ptr<umpire::strategy::AllocationStrategy> >;

template<int FirstFixed, int Current, int LastFixed, int Increment>
struct make_fixed_pool_array {
  static void eval(PoolVector& fixed_pool, Allocator allocator) {
    const int index = Current - FirstFixed;
    const std::size_t size = 1 << Current;
    std::stringstream ss{"internal_fixed_"};
    ss << size;
    fixed_pool[index] = std::make_shared<FixedPool<unsigned char[size]> >(ss.str(), -1, allocator);
    make_fixed_pool_array<FirstFixed, Current+Increment, LastFixed, Increment>::eval(fixed_pool, allocator);
  }
};

template<int FirstFixed, int LastFixed, int Increment>
struct make_fixed_pool_array<FirstFixed,LastFixed,LastFixed,Increment> {
  static void eval(PoolVector& fixed_pool, Allocator allocator) {
    const int index = LastFixed - FirstFixed;
    const std::size_t size = 1 << LastFixed;
    std::stringstream ss{"internal_fixed_"};
    ss << size;
    fixed_pool[index] = std::make_shared<FixedPool<unsigned char[size]> >(ss.str(), -1, allocator);
  }
};
}

template<int FirstFixed, int Increment, int LastFixed>
MixedPoolImpl<FirstFixed,Increment,LastFixed>::MixedPoolImpl(
    const std::string& name,
    int id,
    Allocator allocator) noexcept
:
  AllocationStrategy(name, id),
  m_fixed_pool((LastFixed - FirstFixed)/Increment + 1),
  m_allocator(allocator.getAllocationStrategy())
{
  m_dynamic_pool = std::make_shared<DynamicPool>(
      "internal_dynamic_pool",
      -1,
      allocator);

  make_fixed_pool_array<FirstFixed, FirstFixed, LastFixed, Increment>::eval(m_fixed_pool, allocator);
}


template<int FirstFixed, int Increment, int LastFixed>
void* MixedPoolImpl<FirstFixed,Increment,LastFixed>::allocate(size_t bytes)
{
  size_t nearest = 1;

  size_t original_bytes = bytes;

  if (bytes <= 1) {
    nearest = 1;
  } else {
    nearest = 2;
    bytes--;
    while (bytes >>= 1) nearest <<= 1;
  }

  size_t original_nearest = nearest;
  size_t nearest_index = 0;
  while (nearest >>= 1) nearest_index++;
  nearest_index -= FirstFixed;

  (void) original_bytes;

  //std::cout << nearest << std::endl;

  if (nearest_index < m_fixed_pool.size()) {
    return m_fixed_pool[nearest_index]->allocate(original_nearest);
  }
  else {
    return m_dynamic_pool->allocate(original_bytes);
  }

  return nullptr;
}

template<int FirstFixed, int Increment, int LastFixed>
void MixedPoolImpl<FirstFixed,Increment,LastFixed>::deallocate(void*)
{
}

template<int FirstFixed, int Increment, int LastFixed>
void MixedPoolImpl<FirstFixed,Increment,LastFixed>::release()
{
}

template<int FirstFixed, int Increment, int LastFixed>
long MixedPoolImpl<FirstFixed,Increment,LastFixed>::getCurrentSize() const noexcept
{
  return 0;
}

template<int FirstFixed, int Increment, int LastFixed>
long MixedPoolImpl<FirstFixed,Increment,LastFixed>::getActualSize() const noexcept
{
  return 0;
}

template<int FirstFixed, int Increment, int LastFixed>
long MixedPoolImpl<FirstFixed,Increment,LastFixed>::getHighWatermark() const noexcept
{
  return 0;
}

template<int FirstFixed, int Increment, int LastFixed>
Platform MixedPoolImpl<FirstFixed,Increment,LastFixed>::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

}
}
