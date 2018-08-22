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
#ifndef UMPIRE_DefaultMemoryResource_INL
#define UMPIRE_DefaultMemoryResource_INL

#include "umpire/resource/DefaultMemoryResource.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"

#include <memory>
#include <sstream>

namespace umpire {
namespace resource {

template<typename _allocator>
DefaultMemoryResource<_allocator>::DefaultMemoryResource(Platform platform, const std::string& name, int id, MemoryResourceTraits traits) :
  MemoryResource(name, id, traits),
  m_allocator(),
  m_current_size(0l),
  m_highwatermark(0l),
  m_platform(platform)
{
}

template<typename _allocator>
void* DefaultMemoryResource<_allocator>::allocate(size_t bytes)
{
  void* ptr = m_allocator.allocate(bytes);
  ResourceManager::getInstance().registerAllocation(ptr, new util::AllocationRecord{ptr, bytes, this->shared_from_this()});

  m_current_size += bytes;
  if (m_current_size > m_highwatermark)
    m_highwatermark = m_current_size;

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);

  UMPIRE_RECORD_STATISTIC(getName(), "ptr", reinterpret_cast<uintptr_t>(ptr), "size", bytes, "event", "allocate");

  return ptr;
}

template<typename _allocator>
void DefaultMemoryResource<_allocator>::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");

  UMPIRE_RECORD_STATISTIC(getName(), "ptr", reinterpret_cast<uintptr_t>(ptr), "size", 0x0, "event", "deallocate");

  m_allocator.deallocate(ptr);
  util::AllocationRecord* record = ResourceManager::getInstance().deregisterAllocation(ptr);
  m_current_size -= record->m_size;
  delete record;
}

template<typename _allocator>
long DefaultMemoryResource<_allocator>::getCurrentSize()
{
  UMPIRE_LOG(Debug, "() returning " << m_current_size);
  return m_current_size;
}

template<typename _allocator>
long DefaultMemoryResource<_allocator>::getHighWatermark()
{
  UMPIRE_LOG(Debug, "() returning " << m_highwatermark);
  return m_highwatermark;
}

template<typename _allocator>
Platform DefaultMemoryResource<_allocator>::getPlatform()
{
  return m_platform;
}

} // end of namespace resource
} // end of namespace umpire
#endif // UMPIRE_DefaultMemoryResource_INL
