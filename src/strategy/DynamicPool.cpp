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
#include "umpire/strategy/DynamicPool.hpp"

#include "umpire/ResourceManager.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

DynamicPool::DynamicPool(
    const std::string& name,
    int id,
    Allocator allocator,
    size_t min_alloc_size) :
  AllocationStrategy(name, id),
  dpa(nullptr),
  m_current_size(0),
  m_highwatermark(0),
  m_allocator(allocator.getAllocationStrategy())
{
  dpa = new DynamicPoolAllocator<>(m_allocator, min_alloc_size);
}

void*
DynamicPool::allocate(size_t bytes)
{
  UMPIRE_LOG(Debug, "(bytes=" << bytes << ")");
  void* ptr = dpa->allocate(bytes);
  ResourceManager::getInstance().registerAllocation(ptr, new util::AllocationRecord{ptr, bytes, this->shared_from_this()});

  m_current_size += bytes;
  if (m_current_size > m_highwatermark)
    m_highwatermark = m_current_size;

  return ptr;
}

void 
DynamicPool::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
  dpa->deallocate(ptr);
  m_current_size -= ResourceManager::getInstance().getSize(ptr);

  ResourceManager::getInstance().deregisterAllocation(ptr);
}

long 
DynamicPool::getCurrentSize()
{ 
  return dpa->totalSize(); 
}

long 
DynamicPool::getHighWatermark()
{ 
  UMPIRE_LOG(Debug, "() returning " << m_highwatermark);
  return m_highwatermark;
}

Platform 
DynamicPool::getPlatform()
{ 
  return m_allocator->getPlatform();
}

} // end of namespace strategy
} // end of namespace umpire
