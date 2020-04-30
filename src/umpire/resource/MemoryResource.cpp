//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/MemoryResource.hpp"

namespace umpire {
namespace resource {

MemoryResource::MemoryResource(const std::string& name, int id, MemoryResourceTraits traits) :
  strategy::AllocationStrategy(name, id),
  m_traits(traits)
{
}

MemoryResourceTraits
MemoryResource::getTraits() const noexcept
{
  return m_traits;
}

strategy::AllocationStrategy*
MemoryResource::getAllocationResource() noexcept
{
  return this;
}

} // namespace resource
} // namespace umpire