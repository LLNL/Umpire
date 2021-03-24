//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_NullMemoryResource_INL
#define UMPIRE_NullMemoryResource_INL

#include "umpire/resource/NullMemoryResource.hpp"

#include <memory>
#include <sstream>

#include "umpire/util/Macros.hpp"

#if !defined(_MSC_VER)
#include <sys/mman.h>
#else
#include <windows.h>
#endif

namespace umpire {
namespace resource {

NullMemoryResource::NullMemoryResource(Platform platform,
                                       const std::string& name, int id,
                                       MemoryResourceTraits traits)
    : MemoryResource(name, id, traits), m_platform{platform}, m_size_map{}
{
}

void* NullMemoryResource::allocate(std::size_t bytes)
{
#if 0
#if !defined(_MSC_VER)
  void* ptr{mmap(NULL, bytes, PROT_NONE,
                 (MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE), -1, 0)};
#else
  void* ptr{VirtualAlloc(NULL, bytes, MEM_RESERVE, PAGE_NOACCESS)};
#endif
#else
  // void* ptr = reinterpret_cast<void*>(0x200000070030);
  void* ptr = reinterpret_cast<void*>(0x200000070000);
#endif

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);
  std::cout << "NullMemoryResource: " << bytes << " allocated at " << ptr << std::endl;

  m_size_map.insert(ptr, bytes);

  return ptr;
}

void NullMemoryResource::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");

#if 0
  auto iter = m_size_map.find(ptr);
  auto size = iter->second;
#endif

  m_size_map.erase(ptr);

#if 0
#if !defined(_MSC_VER)
  munmap(ptr, *size);
#else
  VirtualFree(ptr, *size, MEM_RELEASE);
#endif
#endif
}

std::size_t NullMemoryResource::getCurrentSize() const noexcept
{
  return 0;
}

std::size_t NullMemoryResource::getHighWatermark() const noexcept
{
  return 0;
}

bool NullMemoryResource::isAccessibleFrom(Platform p) noexcept
{
  if(p != Platform::undefined)
    UMPIRE_LOG(Debug, "NullMemoryResource: platform is not accessible");
  return false;
}

Platform NullMemoryResource::getPlatform() noexcept
{
  return m_platform;
}

} // end of namespace resource
} // end of namespace umpire
#endif // UMPIRE_NullMemoryResource_INL
