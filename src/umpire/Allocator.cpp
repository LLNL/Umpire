//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////
#include "umpire/Allocator.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/Replay.hpp"

#if defined(UMPIRE_ENABLE_STATISTICS)
#include "umpire/util/StatisticsDatabase.hpp"
#include "umpire/util/Statistic.hpp"
#endif

namespace umpire {

Allocator::Allocator(strategy::AllocationStrategy* allocator) noexcept:
  m_allocator(allocator)
{
}

void*
Allocator::allocate(size_t bytes)
{
  void* ret = nullptr;

  UMPIRE_LOG(Debug, "(" << bytes << ")");

  UMPIRE_REPLAY("allocate_pre" << "," << bytes << "," << m_allocator);

  ret = m_allocator->allocate(bytes);

  UMPIRE_REPLAY("allocate_post" << "," << bytes << "," << m_allocator << "," << ret);

  UMPIRE_RECORD_STATISTIC(getName(), "ptr", reinterpret_cast<uintptr_t>(ret), "size", bytes, "event", "allocate");
  return ret;
}

void
Allocator::deallocate(void* ptr)
{
  UMPIRE_REPLAY("deallocate," << ptr << "," << m_allocator);

  UMPIRE_LOG(Debug, "(" << ptr << ")");

  UMPIRE_RECORD_STATISTIC(getName(), "ptr", reinterpret_cast<uintptr_t>(ptr), "size", 0x0, "event", "deallocate");

  if (!ptr) {
    UMPIRE_LOG(Info, "Deallocating a null pointer");
    return;
  } else {
    m_allocator->deallocate(ptr);
  }
}

void
Allocator::release()
{
  UMPIRE_REPLAY("release," <<  m_allocator);

  UMPIRE_LOG(Debug, "");

  m_allocator->release();
}

size_t
Allocator::getSize(void* ptr) const
{
  UMPIRE_LOG(Debug, "(" << ptr << ")");
  return ResourceManager::getInstance().getSize(ptr);
}

size_t
Allocator::getHighWatermark() const noexcept
{
  return m_allocator->getHighWatermark();
}

size_t
Allocator::getCurrentSize() const noexcept
{
  return m_allocator->getCurrentSize();
}

size_t
Allocator::getActualSize() const noexcept
{
  return m_allocator->getActualSize();
}

const std::string&
Allocator::getName() const noexcept
{
  return m_allocator->getName();
}

int
Allocator::getId() const noexcept
{
  return m_allocator->getId();
}

strategy::AllocationStrategy*
Allocator::getAllocationStrategy() noexcept
{
  UMPIRE_LOG(Debug, "() returning " << m_allocator);
  return m_allocator;
}

Platform
Allocator::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

std::ostream& operator<<(std::ostream& os, const Allocator& allocator) {
    os << *allocator.m_allocator;
    return os;
}

} // end of namespace umpire
