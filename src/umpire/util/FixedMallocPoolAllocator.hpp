//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_FixedMallocPoolAllocator_HPP
#define UMPIRE_FixedMallocPoolAllocator_HPP

#include "umpire/util/FixedMallocPool.hpp"

#include <cstdlib>
#include <new>

namespace umpire {
namespace util {

template <class T>
struct FixedMallocPoolAllocator {
  using value_type = T;

  FixedMallocPoolAllocator() : m_pool{sizeof(T)} {}

  T* allocate(std::size_t n) {
    if(n > std::size_t(-1) / sizeof(T)) throw std::bad_alloc();
    if(auto p = static_cast<T*>(m_pool.allocate())) return p;
    throw std::bad_alloc();
  }
  void deallocate(T* p, std::size_t) noexcept { m_pool.deallocate(p); }

private:
  FixedMallocPool m_pool;
};

template <class T, class U>
bool operator==(const FixedMallocPoolAllocator<T>&,
                const FixedMallocPoolAllocator<U>&)
{
  return true;
}

template <class T, class U>
bool operator!=(const FixedMallocPoolAllocator<T>&,
                const FixedMallocPoolAllocator<U>&)
{
  return false;
}

} // end namespace util
} // end namespace umpire

#endif // UMPIRE_FixedMallocPoolAllocator_HPP
