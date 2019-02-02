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

#include "umpire/resource/NumaMemoryResource.hpp"
#include "umpire/util/Numa.hpp"
#include "umpire/util/Macros.hpp"

#include <cstddef>
#include <numa.h>

namespace umpire {
namespace resource {

NumaMemoryResource::NumaMemoryResource(const std::string& name, int id, MemoryResourceTraits traits) :
  MemoryResource(name, id, traits),
  umpire::strategy::mixins::Inspector(),
  m_platform(Platform::numa)
{
}

void* NumaMemoryResource::allocate(size_t bytes)
{
  void *ptr = nullptr;

  // Need to keep track of allocation sizes, so do this before the
  // allocation, but make sure to keep alignment of the actual alignment
  ptr = numa::allocate_on_node(bytes, m_traits.numa_node);

  registerAllocation(ptr, bytes, this->shared_from_this());

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);
  UMPIRE_RECORD_STATISTIC(getName(), "ptr", reinterpret_cast<uintptr_t>(ptr), "size", bytes, "event", "allocate");

  return ptr;
}

void NumaMemoryResource::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");

  UMPIRE_RECORD_STATISTIC(getName(), "ptr", reinterpret_cast<uintptr_t>(ptr), "size", 0x0, "event", "deallocate");

  numa::deallocate(ptr);

  deregisterAllocation(ptr);
}

long NumaMemoryResource::getCurrentSize() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << m_current_size);
  return m_current_size;
}

long NumaMemoryResource::getHighWatermark() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << m_high_watermark);
  return m_high_watermark;
}

Platform NumaMemoryResource::getPlatform() noexcept
{
  return Platform::numa;
}

} // end of namespace resource
} // end of namespace umpire
