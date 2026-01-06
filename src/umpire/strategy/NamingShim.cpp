//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/NamingShim.hpp"

#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace strategy {

NamingShim::NamingShim(const std::string& name, int id, Allocator allocator)
    : AllocationStrategy{name, id, allocator.getAllocationStrategy(), "NamingShim"},
      m_counter{0},
      m_allocator(allocator.getAllocationStrategy())
{
}

NamingShim::~NamingShim()
{
}

void* NamingShim::allocate(std::size_t bytes)
{
  std::string name{m_name + "_alloc_" + std::to_string(m_counter++)};
  return m_allocator->allocate_named_internal(name, bytes);
}

void NamingShim::deallocate(void* ptr, std::size_t UMPIRE_UNUSED_ARG(size))
{
  m_allocator->deallocate_internal(ptr);
}

Platform NamingShim::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

MemoryResourceTraits NamingShim::getTraits() const noexcept
{
  return m_allocator->getTraits();
}

} // end of namespace strategy
} // end of namespace umpire
