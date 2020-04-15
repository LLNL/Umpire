//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_MpiSharedMemoryResource_INL
#define UMPIRE_MpiSharedMemoryResource_INL

#include "umpire/resource/MpiSharedMemoryResource.hpp"

#include "umpire/util/Macros.hpp"

#include <memory>
#include <sstream>

namespace umpire {
namespace resource {

MpiSharedMemoryResource::MpiSharedMemoryResource(
    Platform platform, 
    const std::string& name,
    int id,
    MemoryResourceTraits traits) :
    MemoryResource(name, id, traits)
  , m_platform{platform}
{
}

void* MpiSharedMemoryResource::allocate(std::size_t bytes)
{
  void* ptr = (void*)0xdeadbeaf0000;
  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);
  UMPIRE_RECORD_STATISTIC(getName(), "ptr", reinterpret_cast<uintptr_t>(ptr), "size", bytes, "event", "allocate");

  return ptr;
}

void MpiSharedMemoryResource::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
  UMPIRE_RECORD_STATISTIC(getName(), "ptr", reinterpret_cast<uintptr_t>(ptr), "size", 0x0, "event", "deallocate");
}

std::size_t MpiSharedMemoryResource::getCurrentSize() const noexcept
{
  return 0;
}

std::size_t MpiSharedMemoryResource::getHighWatermark() const noexcept
{
  return 0;
}

Platform MpiSharedMemoryResource::getPlatform() noexcept
{
  return m_platform;
}

} // end of namespace resource
} // end of namespace umpire
#endif // UMPIRE_MpiSharedMemoryResource_INL
