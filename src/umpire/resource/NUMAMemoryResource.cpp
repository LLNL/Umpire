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

#include "umpire/resource/NUMAMemoryResource.hpp"
#include "umpire/util/Macros.hpp"

#if defined(UMPIRE_ENABLE_NUMA)

#include <cstddef>
#include <numa.h>

union alloc_or_size_t {
  std::size_t bytes;
  std::max_align_t a;
};

#endif

namespace umpire {
namespace resource {

NUMAMemoryResource::NUMAMemoryResource(const std::string& name, int id, MemoryResourceTraits traits) :
  MemoryResource(name, id, traits),
  umpire::strategy::mixins::Inspector(),
  m_platform(Platform::cpu)
{
#if defined(UMPIRE_ENABLE_NUMA)
  if (numa_available() < 0) UMPIRE_ERROR("libnuma is not usable.");
#else
  UMPIRE_LOG(Warning, "NUMA is not available. Falling back to ::malloc()");
#endif
}

void* NUMAMemoryResource::allocate(size_t bytes)
{
  void *ptr = nullptr;
#if defined(UMPIRE_ENABLE_NUMA)
  // Need to keep track of allocation sizes, so do this before the
  // allocation, but make sure to keep alignment of the actual alignment
  union alloc_or_size_t* s = static_cast<union alloc_or_size_t*>(
    numa_alloc_onnode(sizeof(*s) + bytes, m_traits.numa_node));
  if (s) {
    s->bytes = bytes;
    ptr = ++s;
  }
  else {
    UMPIRE_ERROR("numa_alloc_onnode( bytes = " << sizeof(*s) + bytes << ", " << m_traits.numa_node << " ) failed");
  }
#else
  void* s = ::malloc(bytes);
  if (s) {
    ptr = s;
  }
  else {
    UMPIRE_ERROR("malloc( bytes = " << bytes << " ) failed");
  }
#endif

  registerAllocation(ptr, bytes, this->shared_from_this());

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);
  UMPIRE_RECORD_STATISTIC(getName(), "ptr", reinterpret_cast<uintptr_t>(ptr), "size", bytes, "event", "allocate");

  return ptr;
}

void NUMAMemoryResource::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");

  UMPIRE_RECORD_STATISTIC(getName(), "ptr", reinterpret_cast<uintptr_t>(ptr), "size", 0x0, "event", "deallocate");

  if (ptr) {
#if defined(UMPIRE_ENABLE_NUMA)
    union alloc_or_size_t* s = static_cast<union alloc_or_size_t*>(ptr);
    s--;
    numa_free(s, sizeof(*s) + s->bytes);
#else
    ::free(ptr);
#endif
  }

  deregisterAllocation(ptr);
}

long NUMAMemoryResource::getCurrentSize() noexcept
{
  UMPIRE_LOG(Debug, "() returning " << m_current_size);
  return m_current_size;
}

long NUMAMemoryResource::getHighWatermark() noexcept
{
  UMPIRE_LOG(Debug, "() returning " << m_high_watermark);
  return m_high_watermark;
}

Platform NUMAMemoryResource::getPlatform() noexcept
{
  return Platform::cpu;
}

} // end of namespace resource
} // end of namespace umpire
