//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/strategy/AlignedAllocator.hpp"

#include "umpire/config.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

AlignedAllocator::AlignedAllocator(const std::string& name, int id, Allocator allocator, std::size_t alignment)
    : AllocationStrategy{name, id, allocator.getAllocationStrategy(), "AlignedAllocator"},
      m_allocator(allocator.getAllocationStrategy()),
      m_alignment{alignment},
      m_mask{static_cast<uintptr_t>(~(m_alignment - 1))}
{
  if (m_allocator->getPlatform() != Platform::host) {
    UMPIRE_ERROR(runtime_error, "Cannot construct AlignedAllocator from non-host Allocator.");
  }

  if (!(m_alignment >= 16 && ((m_alignment & (m_alignment - 1)) == 0))) {
    UMPIRE_ERROR(runtime_error,
                 "AlignedAllocator alignment must be a power of 2 greater than or equal "
                 "to 16");
  }
}

void* AlignedAllocator::allocate(std::size_t bytes)
{
#ifdef UMPIRE_ENABLE_HEADER_INTROSPECTION
  // With header introspection, the Allocator layer adds a 64-byte header BEFORE the pointer
  // we return. So if alignment > 64, we need to ensure the final user pointer (our_return + 64)
  // is aligned. We do this by over-allocating and returning a pointer where (ptr + 64) is aligned.
  constexpr std::size_t header_size = 64;

  if (m_alignment > header_size) {
    // Need extra space to align the user pointer (which will be header_size bytes after our return)
    std::size_t total_bytes = bytes + sizeof(void*) + m_alignment + header_size;
    UMPIRE_LOG(Debug, "requested: " << bytes << " actual: " << total_bytes << " (with header compensation)");

    uintptr_t ptr{reinterpret_cast<uintptr_t>(m_allocator->allocate_internal(total_bytes))};

    // Calculate where the user pointer will be (header_size bytes after our return)
    // We want (aligned_ptr + header_size) to be aligned
    uintptr_t target_aligned = ((ptr + sizeof(void*) + header_size + (m_alignment - 1)) & m_mask);
    uintptr_t aligned_ptr = target_aligned - header_size;

    uintptr_t* header = (uintptr_t*)(aligned_ptr - sizeof(void*));
    *header = ptr;

    UMPIRE_LOG(Debug, "ptr: " << reinterpret_cast<void*>(ptr)
                      << " aligned: " << reinterpret_cast<void*>(aligned_ptr)
                      << " user_ptr: " << reinterpret_cast<void*>(target_aligned));
    return reinterpret_cast<void*>(aligned_ptr);
  }
#endif

  // Original behavior: without headers or when alignment <= header_size
  std::size_t total_bytes = bytes + sizeof(void*) + m_alignment - 1;
  UMPIRE_LOG(Debug, "requested: " << bytes << " actual: " << bytes + m_alignment - 1);

  uintptr_t ptr{reinterpret_cast<uintptr_t>(m_allocator->allocate_internal(total_bytes))};
  uintptr_t aligned_ptr{static_cast<uintptr_t>((ptr + sizeof(void*) + (m_alignment - 1)) & m_mask)};
  uintptr_t* header = (uintptr_t*)(aligned_ptr - sizeof(void*));
  *header = ptr;

  UMPIRE_LOG(Debug, "ptr: " << reinterpret_cast<void*>(ptr) << " aligned: " << reinterpret_cast<void*>(aligned_ptr));
  return reinterpret_cast<void*>(aligned_ptr);
}

void AlignedAllocator::deallocate(void* ptr, std::size_t size)
{
  uintptr_t aligned_ptr{reinterpret_cast<uintptr_t>(ptr)};
  uintptr_t* header = (uintptr_t*)(aligned_ptr - sizeof(void*));
  void* base_ptr = reinterpret_cast<void*>(*header);

  UMPIRE_LOG(Debug, "ptr: " << reinterpret_cast<void*>(ptr) << " base_ptr: " << reinterpret_cast<void*>(base_ptr));

#ifdef UMPIRE_ENABLE_HEADER_INTROSPECTION
  constexpr std::size_t header_size = 64;
  if (m_alignment > header_size) {
    // Match the allocation size from allocate()
    std::size_t total_bytes = size + sizeof(void*) + m_alignment + header_size;
    return m_allocator->deallocate_internal(base_ptr, total_bytes);
  }
#endif

  // Original behavior
  std::size_t total_bytes = size + sizeof(void*) + m_alignment - 1;
  return m_allocator->deallocate_internal(base_ptr, total_bytes);
}

Platform AlignedAllocator::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

MemoryResourceTraits AlignedAllocator::getTraits() const noexcept
{
  return m_allocator->getTraits();
}

} // end of namespace strategy
} // end of namespace umpire
