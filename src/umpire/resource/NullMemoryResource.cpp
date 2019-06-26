//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_NullMemoryResource_INL
#define UMPIRE_NullMemoryResource_INL

#include "umpire/resource/NullMemoryResource.hpp"

#include "umpire/util/Macros.hpp"

#include <memory>
#include <sstream>

#include <sys/mman.h>

namespace umpire {
namespace resource {

NullMemoryResource::NullMemoryResource(
    Platform platform, 
    const std::string& name,
    int id,
    MemoryResourceTraits traits) :
  MemoryResource(name, id, traits),
  m_platform(platform)
{
}

void* NullMemoryResource::allocate(std::size_t bytes)
{
  void* ptr{mmap(NULL, bytes, PROT_NONE, (MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE), -1, 0)};

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);
  UMPIRE_RECORD_STATISTIC(getName(), "ptr", reinterpret_cast<uintptr_t>(ptr), "size", bytes, "event", "allocate");

  return ptr;
}

void NullMemoryResource::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
  UMPIRE_RECORD_STATISTIC(getName(), "ptr", reinterpret_cast<uintptr_t>(ptr), "size", 0x0, "event", "deallocate");

  munmap(ptr, 4096);
}

std::size_t NullMemoryResource::getCurrentSize() const noexcept
{
  return 0;
}

std::size_t NullMemoryResource::getHighWatermark() const noexcept
{
  return 0;
}

Platform NullMemoryResource::getPlatform() noexcept
{
  return m_platform;
}

} // end of namespace resource
} // end of namespace umpire
#endif // UMPIRE_NullMemoryResource_INL
