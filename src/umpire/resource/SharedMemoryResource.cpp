//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/SharedMemoryResource.hpp"

namespace umpire {
namespace resource {

SharedMemoryResource::SharedMemoryResource(const std::string& name, int id, MemoryResourceTraits traits) :
  MemoryResource{name, id, traits},
  strategy::SharedAllocationStrategy{name, id},
  m_traits{traits}
{
}

MemoryResourceTraits
SharedMemoryResource::getTraits() const noexcept
{
  return m_traits;
}

} // end of namespace resource
} // end of namespace umpire
