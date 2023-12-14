//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/StreamAwareAllocator.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

StreamAwareAllocator::StreamAwareAllocator(const std::string& name, int id, Allocator allocator)
    : AllocationStrategy{name, id, allocator.getAllocationStrategy(), "StreamAwareAllocator"},
      m_allocator(allocator.getAllocationStrategy())
{
}

void* StreamAwareAllocator::allocate(void* stream, std::size_t bytes)
{
  unsigned int size = m_registered_streams.size();
  UMPIRE_LOG(Debug, "Size of registered streams vector is: " << size);
  
  for (unsigned int i = 0; i < size; i++) {
    if (m_registered_streams.at(i) == stream) {
      UMPIRE_LOG(Debug, "I found a registered stream, I am allocating bytes:  " << bytes );
      return allocate(bytes);
      //allocate(bytes);    
    }
  }
    
  UMPIRE_LOG(Debug, "I did not find a registered stream so I am adding it to vector.");
  m_registered_streams.push_back(stream);
  return allocate(bytes);
}

void StreamAwareAllocator::deallocate(void* stream, void* ptr, std::size_t size)
{
  m_allocator->deallocate_internal(ptr, size);
}

void* StreamAwareAllocator::allocate(std::size_t bytes)
{
  return m_allocator->allocate_internal(bytes);
}

void StreamAwareAllocator::deallocate(void* ptr, std::size_t size)
{
  m_allocator->deallocate_internal(ptr, size);
}

Platform StreamAwareAllocator::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

MemoryResourceTraits StreamAwareAllocator::getTraits() const noexcept
{
  return m_allocator->getTraits();
}

} // end of namespace strategy
} // end of namespace umpire
