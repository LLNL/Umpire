//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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

#if defined(UMPIRE_ENABLE_STATISTICS)
#include "umpire/util/StatisticsDatabase.hpp"
#include "umpire/util/Statistic.hpp"
#endif

namespace umpire {

Allocator::Allocator(std::shared_ptr<strategy::AllocationStrategy> allocator):
  m_allocator(allocator),
  m_mutex(new std::mutex())
{
}

void*
Allocator::allocate(size_t bytes)
{
  void* ret = nullptr;
  try {
    UMPIRE_LOCK;

    UMPIRE_LOG(Debug, "(" << bytes << ")");
    ret = m_allocator->allocate(bytes);

    UMPIRE_RECORD_STATISTIC(getName(), "ptr", reinterpret_cast<uintptr_t>(ret), "size", bytes, "event", "allocate");

    UMPIRE_UNLOCK;
  } catch (...) {
    UMPIRE_UNLOCK;
    throw;
  }
  return ret;
}

void
Allocator::deallocate(void* ptr)
{
  try {
    UMPIRE_LOCK;

    UMPIRE_ASSERT("Deallocate called with nullptr" && ptr);
    UMPIRE_LOG(Debug, "(" << ptr << ")");

    UMPIRE_RECORD_STATISTIC(getName(), "ptr", reinterpret_cast<uintptr_t>(ptr), "size", 0x0, "event", "deallocate");

    m_allocator->deallocate(ptr);
    UMPIRE_UNLOCK;
  } catch (...) {
    UMPIRE_UNLOCK;
    throw;
  }
}

size_t
Allocator::getSize(void* ptr)
{
  UMPIRE_LOG(Debug, "(" << ptr << ")");
  return ResourceManager::getInstance().getSize(ptr);
}

size_t
Allocator::getHighWatermark()
{
  return m_allocator->getHighWatermark();
}

size_t
Allocator::getCurrentSize()
{
  return m_allocator->getCurrentSize();
}

size_t
Allocator::getActualSize()
{
  return m_allocator->getActualSize();
}

std::string
Allocator::getName()
{
  return m_allocator->getName();
}

int
Allocator::getId()
{
  return m_allocator->getId();
}

std::shared_ptr<strategy::AllocationStrategy>
Allocator::getAllocationStrategy()
{
  UMPIRE_LOG(Debug, "() returning " << m_allocator);
  return m_allocator;
}

Platform
Allocator::getPlatform()
{
  return m_allocator->getPlatform();
}

} // end of namespace umpire
