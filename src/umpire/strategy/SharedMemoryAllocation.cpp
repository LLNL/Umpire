//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/strategy/SharedMemoryAllocation.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

SharedMemoryAllocation::SharedMemoryAllocation(const std::string& name, int id) noexcept :
  AllocationStrategy{name, id}
{
}

void*
SharedMemoryAllocation::allocate(std::size_t UMPIRE_UNUSED_ARG(bytes))
{
  UMPIRE_ERROR("Unnamed shared memory allocation is not supported")
  return nullptr;
}

void*
SharedMemoryAllocation::allocate(std::string UMPIRE_UNUSED_ARG(name), std::size_t UMPIRE_UNUSED_ARG(bytes))
{
  UMPIRE_ERROR("Named shared memory allocation is not supported")
  return nullptr;
}

void*
SharedMemoryAllocation::get_allocation_by_name(std::string UMPIRE_UNUSED_ARG(allocation_name))
{
  UMPIRE_ERROR("Named shared memory allocation is not supported")
  return nullptr;
}

} // end of namespace strategy
} // end of namespace umpire
