//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/strategy/AlignedAllocator.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

AlignedAllocator::AlignedAllocator(
    const std::string& name,
    int id,
    Allocator allocator,
    std::size_t alignment) :
  AllocationStrategy(name, id),
  m_allocator(allocator.getAllocationStrategy()),
  m_alignment{alignment},
  m_mask{static_cast<uintptr_t>(~(m_alignment-1))}
{
  if (m_allocator->getPlatform() != Platform::host) {
    UMPIRE_ERROR("Cannot construct AlignedAllocator from non-host Allocator.");
  }
}

void* 
AlignedAllocator::allocate(std::size_t bytes)
{
  std::size_t total_bytes = bytes+sizeof(void*)+m_alignment-1;
  UMPIRE_LOG(Debug, "requested: " << bytes << " actual: " << bytes+m_alignment-1);

  uintptr_t ptr{reinterpret_cast<uintptr_t>(m_allocator->allocate(total_bytes))};
  uintptr_t aligned_ptr{static_cast<uintptr_t>((reinterpret_cast<uintptr_t>(ptr) + sizeof(void*) + (m_alignment-1)) & m_mask)}; 
  uintptr_t* header = (uintptr_t*) (aligned_ptr - sizeof(void*));
  *header = ptr;

  std::cout << ptr << std::endl;
  std::cout << aligned_ptr << std::endl;

  UMPIRE_LOG(Debug, "ptr: " << reinterpret_cast<void*>(ptr) << " aligned: " << reinterpret_cast<void*>(aligned_ptr));
  return reinterpret_cast<void*>(aligned_ptr);
}

void 
AlignedAllocator::deallocate(void* ptr)
{
  uintptr_t aligned_ptr{reinterpret_cast<uintptr_t>(ptr)};
  uintptr_t* header = (uintptr_t*) (aligned_ptr - sizeof(void*));
  void* base_ptr = reinterpret_cast<void*>(*header);
  
  UMPIRE_LOG(Debug, "ptr: " << reinterpret_cast<void*>(ptr) << " base_ptr: " << reinterpret_cast<void*>(base_ptr));
  return m_allocator->deallocate(base_ptr);
}

Platform 
AlignedAllocator::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

MemoryResourceTraits
AlignedAllocator::getTraits() const noexcept
{
  return m_allocator->getTraits();
}

} // end of namespace strategy
} // end of namespace umpire
