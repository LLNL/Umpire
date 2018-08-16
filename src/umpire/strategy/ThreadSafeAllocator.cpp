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
  m_current_size(0),
  m_highwatermark(0),
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

    m_current_size += bytes;
    if (m_current_size > m_highwatermark)
      m_highwatermark = m_current_size;


    UMPIRE_UNLOCK;
  } catch (...) {
    UMPIRE_UNLOCK;
    throw;
  }

  ResourceManager::getInstance().registerAllocation(ret, 
      new util::AllocationRecord{ret, bytes, this->shared_from_this()});

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

  util::AllocationRecord* record = ResourceManager::getInstance().deregisterAllocation(ptr);
  m_current_size -= record->m_size;

  delete record;
}

long
ThreadSafeAllocator::getCurrentSize()
{
  long size;
  
  size = m_current_size;

  return size;
}

long
ThreadSafeAllocator::getHighWatermark()
{
  long size;
  
  size = m_highwatermark;

  return size;
}


Platform
ThreadSafeAllocator::getPlatform()
{

  return m_allocator->getPlatform();
}

} // end of namespace strategy
} // end of namespace umpire
