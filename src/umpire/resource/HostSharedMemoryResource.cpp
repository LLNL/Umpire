//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/resource/HostSharedMemoryResource.hpp"
#include "umpire/resource/HostSharedMemoryResourceImpl.hpp"
#include "umpire/resource/MemoryResource.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace resource {

HostSharedMemoryResource::HostSharedMemoryResource(Platform platform,
                                       const std::string& name, int id,
                                       MemoryResourceTraits traits)
    :
      MemoryResource{name, id, traits},
      m_platform{platform},
      pimpl{ new impl{ name, traits.size } }
{
}

HostSharedMemoryResource::~HostSharedMemoryResource()
{
}

void* HostSharedMemoryResource::allocate(std::size_t UMPIRE_UNUSED_ARG(bytes))
{
  UMPIRE_ERROR("Shared memory allocation without name is not supported");
}

void* HostSharedMemoryResource::allocate_named(const std::string& name, std::size_t bytes)
{
  void* ptr{ pimpl->allocate_named(name, bytes) };

  UMPIRE_LOG(Debug, "(name=\"" << name << ", requested_size=" << bytes << ") returning: " << ptr);
  return ptr;
}

void HostSharedMemoryResource::deallocate(void* ptr, std::size_t)
{
  return pimpl->deallocate(ptr);
}

bool HostSharedMemoryResource::isAccessibleFrom(Platform p) noexcept
{
  // Todo: Need to determine how to update tests which require
  // unnamed alloc to test.
  //
  UMPIRE_USE_VAR(p);
  return false;
}

Platform HostSharedMemoryResource::getPlatform() noexcept
{
  return m_platform;
}

std::size_t HostSharedMemoryResource::getActualSize() const noexcept
{
  return pimpl->getActualSize();
}

void* HostSharedMemoryResource::find_pointer_from_name(std::string name)
{
  return pimpl->find_pointer_from_name(name);
}
} // end of namespace resource
} // end of namespace umpire
