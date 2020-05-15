//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/NamedSharedStrategy.hpp"
#include "umpire/SharedMemoryAllocator.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

NamedSharedStrategy::NamedSharedStrategy( const std::string& name, int id, SharedMemoryAllocator allocator) noexcept
  : SharedAllocationStrategy{name, id}, m_allocator{allocator.getAllocationStrategy()}
{
}

NamedSharedStrategy::~NamedSharedStrategy()
{
}

void* NamedSharedStrategy::allocate(std::string name, std::size_t bytes)
{
  UMPIRE_LOG(Debug, "(" << name << ", " << bytes << ")");

  void* ret = m_allocator->allocate(name, bytes);

  m_name_to_pointer[name] = ret;
  m_pointer_to_name[ret] = name;

  return ret;
}

void* NamedSharedStrategy::get_allocation_by_name(std::string name)
{
  UMPIRE_LOG(Debug, "(" << name << ")");

  return m_name_to_pointer[name];
}

void NamedSharedStrategy::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "(" << ptr << ")");

  m_name_to_pointer.erase(m_pointer_to_name[ptr]);
  m_pointer_to_name.erase(ptr);
  m_allocator->deallocate(ptr);
}

Platform NamedSharedStrategy::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

} // end of namespace strategy
} // end namespace umpire
