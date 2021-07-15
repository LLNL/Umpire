//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_AllocatorBase_INL
#define UMPIRE_AllocatorBase_INL

#include "umpire/Allocator.hpp"
#include "umpire/config.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {

template <>
inline void* AllocatorBase<Tracked>::allocate(std::size_t bytes)
{
  void* ret = nullptr;

  umpire_ver_5_found = 0;

  UMPIRE_LOG(Debug, "(" << bytes << ")");

  if (0 == bytes) {
    ret = allocateNull();
  } else {
    ret = m_allocator->allocate(bytes);
  }

  registerAllocation(ret, bytes, m_allocator);

  return ret;
}

template <>
inline void AllocatorBase<Tracked>::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "(" << ptr << ")");

  if (!ptr) {
    UMPIRE_LOG(Info,
               "Deallocating a null pointer (This behavior is intentionally "
               "allowed and ignored)");
    return;
  } else {
    auto record = deregisterAllocation(ptr, m_allocator);
    if (!deallocateNull(ptr)) {
      m_allocator->deallocate(ptr, record.size);
    }
  }
}

template <>
inline void* AllocatorBase<Untracked>::allocate(std::size_t bytes)
{
  void* ret = nullptr;

  umpire_ver_5_found = 0;

  UMPIRE_LOG(Debug, "(" << bytes << ")");

  if (0 == bytes) {
    ret = allocateNull();
  } else {
    ret = m_allocator->allocate(bytes);
  }

  return ret;
}

template <>
inline void AllocatorBase<Untracked>::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "(" << ptr << ")");

  if (!ptr) {
    UMPIRE_LOG(Info,
               "Deallocating a null pointer (This behavior is intentionally "
               "allowed and ignored)");
    return;
  } else {
    if (!deallocateNull(ptr)) {
      m_allocator->deallocate(ptr);
    }
  }
}

template <typename Tracking>
AllocatorBase<Tracking>::AllocatorBase(strategy::AllocationStrategy* allocator) noexcept
    : strategy::mixins::Inspector{}, strategy::mixins::AllocateNull{}, m_allocator{allocator}, m_tracking{true}
{
}

template <typename Tracking>
void AllocatorBase<Tracking>::release()
{
  UMPIRE_LOG(Debug, "");

  m_allocator->release();
}

template <>
inline std::size_t AllocatorBase<Tracked>::getSize(void* ptr) const
{
  UMPIRE_LOG(Debug, "(" << ptr << ")");
  // return ResourceManager::getInstance().getSize(ptr);
  return 0;
}

template <>
inline std::size_t AllocatorBase<Untracked>::getSize(void* ptr) const
{
  UMPIRE_LOG(Debug, "(" << ptr << ")");
  return 0;
}

template <typename Tracking>
std::size_t AllocatorBase<Tracking>::getHighWatermark() const noexcept
{
  return m_allocator->getHighWatermark();
}

template <typename Tracking>
std::size_t AllocatorBase<Tracking>::getCurrentSize() const noexcept
{
  return m_allocator->getCurrentSize();
}

template <typename Tracking>
std::size_t AllocatorBase<Tracking>::getActualSize() const noexcept
{
  return m_allocator->getActualSize();
}

template <typename Tracking>
std::size_t AllocatorBase<Tracking>::getAllocationCount() const noexcept
{
  return m_allocator->getAllocationCount();
}

template <typename Tracking>
const std::string& AllocatorBase<Tracking>::getName() const noexcept
{
  return m_allocator->getName();
}

template <typename Tracking>
int AllocatorBase<Tracking>::getId() const noexcept
{
  return m_allocator->getId();
}

template <typename Tracking>
strategy::AllocationStrategy* AllocatorBase<Tracking>::getParent() const noexcept
{
  return m_allocator->getParent();
}

template <typename Tracking>
strategy::AllocationStrategy* AllocatorBase<Tracking>::getAllocationStrategy() noexcept
{
  UMPIRE_LOG(Debug, "() returning " << m_allocator);
  return m_allocator;
}

template <typename Tracking>
Platform AllocatorBase<Tracking>::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

template <typename Tracking>
std::ostream& operator<<(std::ostream& out, AllocatorBase<Tracking>& alloc)
{
  out << alloc.getName();
  return out;
}

} // end of namespace umpire

#endif // UMPIRE_AllocatorBase_INL
