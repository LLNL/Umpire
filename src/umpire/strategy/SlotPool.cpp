//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/SlotPool.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

SlotPool::SlotPool(const std::string& name, int id, Allocator allocator,
                   std::size_t slots)
    : AllocationStrategy(name, id),
      m_current_size(0),
      m_highwatermark(0),
      m_slots(slots),
      m_allocator(allocator.getAllocationStrategy())
{
  UMPIRE_LOG(Debug, "Creating " << m_slots << "-slot pool.");

  m_lengths = new int64_t[m_slots];
  m_pointers = new void*[m_slots];

  for (std::size_t i = 0; i < m_slots; ++i) {
    m_pointers[i] = nullptr;
    m_lengths[i] = 0;
  }
}

SlotPool::~SlotPool()
{
  for (std::size_t i = 0; i < m_slots; ++i) {
    if (m_pointers[i]) {
      m_allocator->deallocate(m_pointers[i]);
      m_pointers[i] = nullptr;
      m_lengths[i] = 0;
    }
  }

  delete[] m_lengths;
  delete[] m_pointers;
}

void* SlotPool::allocate(std::size_t bytes)
{
  void* ptr = nullptr;
  int64_t int_bytes = static_cast<int64_t>(bytes);

  if (int_bytes < 0) {
    UMPIRE_ERROR("allocation request of size: "
                 << bytes << " bytes is too large for this pool");
  }

  for (std::size_t i = 0; i < m_slots; ++i) {
    if (m_lengths[i] == int_bytes) {
      m_lengths[i] = -m_lengths[i];
      ptr = m_pointers[i];
      break;
    } else if (m_lengths[i] == 0) {
      m_lengths[i] = -int_bytes;
      m_pointers[i] = m_allocator->allocate(bytes);
      ptr = m_pointers[i];
      break;
    }
  }

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);
  return ptr;
}

void SlotPool::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
  for (std::size_t i = 0; i < m_slots; ++i) {
    if (m_pointers[i] == ptr) {
      m_lengths[i] = -m_lengths[i];
      ptr = nullptr;
      break;
    }
  }
}

std::size_t SlotPool::getCurrentSize() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << m_current_size);
  return m_current_size;
}

std::size_t SlotPool::getHighWatermark() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << m_highwatermark);
  return m_highwatermark;
}

Platform SlotPool::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

MemoryResourceTraits SlotPool::getTraits() const noexcept
{
  return m_allocator->getTraits();
}

} // end of namespace strategy
} // end of namespace umpire
