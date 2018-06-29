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
#include "umpire/strategy/MonotonicAllocationStrategy.hpp"
#include "umpire/util/Macros.hpp"

#include "umpire/ResourceManager.hpp"

namespace umpire {

namespace strategy {

MonotonicAllocationStrategy::MonotonicAllocationStrategy(
    const std::string& name,
    int id,
    size_t capacity,
    Allocator allocator) :
  AllocationStrategy(name, id),
  m_size(0),
  m_capacity(capacity),
  m_allocator(allocator.getAllocationStrategy())
{
  m_block = m_allocator->allocate(m_capacity);
}

void* 
MonotonicAllocationStrategy::allocate(size_t bytes)
{
  void* ret = static_cast<char*>(m_block) + bytes;
  m_size += bytes;

  if (m_size > m_capacity) {
    UMPIRE_ERROR("MonoticAllocationStrategy capacity exceeded " << m_size << " > " << m_capacity);
  }

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ret);

  ResourceManager::getInstance().registerAllocation(ret, new util::AllocationRecord{ret, bytes, this->shared_from_this()});

  return ret;
}

void 
MonotonicAllocationStrategy::deallocate(void* ptr)
{
  UMPIRE_LOG(Info, "() doesn't do anything");
  // no op

  ResourceManager::getInstance().deregisterAllocation(ptr);
}

long 
MonotonicAllocationStrategy::getCurrentSize()
{
  UMPIRE_LOG(Debug, "() returning " << m_size);
  return m_size;
}

long 
MonotonicAllocationStrategy::getHighWatermark()
{
  UMPIRE_LOG(Debug, "() returning " << m_capacity);
  return m_capacity;
}

Platform 
MonotonicAllocationStrategy::getPlatform()
{
  return m_allocator->getPlatform();
}

} // end of namespace strategy
} // end of namespace umpire
