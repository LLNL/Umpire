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

#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/ThreadSafeAllocator.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

ThreadSafeAllocator::ThreadSafeAllocator(
    const std::string& name,
    int id,
    Allocator allocator) :
  AllocationStrategy(name, id),
  m_allocator(allocator.getAllocationStrategy()),
  m_mutex(new std::mutex())
{
}

void*
ThreadSafeAllocator::allocate(size_t bytes)
{
  void* ret = nullptr;

  try {
    UMPIRE_LOCK;

    ret = m_allocator->allocate(bytes);

    UMPIRE_UNLOCK;
  } catch (...) {
    UMPIRE_UNLOCK;
    throw;
  }

  return ret;
}

void
ThreadSafeAllocator::deallocate(void* ptr)
{
  try {
    UMPIRE_LOCK;

    m_allocator->deallocate(ptr);

    UMPIRE_UNLOCK;
  } catch (...) {
    UMPIRE_UNLOCK;
    throw;
  }
}

long
ThreadSafeAllocator::getCurrentSize() noexcept
{
  return 0;
}

long
ThreadSafeAllocator::getHighWatermark() noexcept
{
  return 0;
}

Platform
ThreadSafeAllocator::getPlatform() noexcept
{

  return m_allocator->getPlatform();
}

} // end of namespace strategy
} // end of namespace umpire
