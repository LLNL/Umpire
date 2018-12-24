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
#include "umpire/Replay.hpp"

#if defined(UMPIRE_ENABLE_STATISTICS)
#include "umpire/util/StatisticsDatabase.hpp"
#include "umpire/util/Statistic.hpp"
#endif

#include "umpire/strategy/SizeLimiter.hpp"

namespace umpire {

Allocator::Allocator(std::shared_ptr<strategy::AllocationStrategy> allocator) noexcept:
  m_allocator(allocator)
{
}

void*
Allocator::allocate(size_t bytes)
{
  void* ret = nullptr;

  UMPIRE_LOG(Debug, "(" << bytes << ")");

  UMPIRE_REPLAY( "allocate," << bytes << "," << m_allocator);
  ret = m_allocator->allocate(bytes);
  UMPIRE_REPLAY_CONT( ret << "\n");

  UMPIRE_RECORD_STATISTIC(getName(), "ptr", reinterpret_cast<uintptr_t>(ret), "size", bytes, "event", "allocate");
  return ret;
}

void
Allocator::deallocate(void* ptr)
{
  UMPIRE_REPLAY( "deallocate," << ptr << "," << m_allocator << "\n");

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
  UMPIRE_REPLAY("release," <<  m_allocator << "\n");

  UMPIRE_LOG(Debug, "");

  m_allocator->release();
}

size_t
Allocator::getSize(void* ptr)
{
  UMPIRE_LOG(Debug, "(" << ptr << ")");
  return ResourceManager::getInstance().getSize(ptr);
}

size_t
Allocator::getHighWatermark() noexcept
{
  return m_allocator->getHighWatermark();
}

size_t
Allocator::getCurrentSize() noexcept
{
  return m_allocator->getCurrentSize();
}

size_t
Allocator::getActualSize() noexcept
{
  return m_allocator->getActualSize();
}

std::string
Allocator::getName() const noexcept
{
  return m_allocator->getName();
}

int
Allocator::getId() const noexcept
{
  return m_allocator->getId();
}

std::shared_ptr<strategy::AllocationStrategy>
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

AllocatorBuilder::AllocatorBuilder(ResourceManager &rm) :
  m_rm(rm), m_strategy(m_rm.getDefaultAllocator().getAllocationStrategy())
{
}

AllocatorBuilder& AllocatorBuilder::onMemoryResource(resource::MemoryResourceType rt)
{
  m_strategy = m_rm.getAllocator(rt).getAllocationStrategy();
  return *this;
}

AllocatorBuilder& AllocatorBuilder::withMaximumSize(const int size_limit)
{
  m_strategy = std::make_shared<strategy::SizeLimiter>(m_strategy->getName(),
                                                       m_strategy->getId(),
                                                       Allocator(m_strategy),
                                                       size_limit);
  return *this;
}

AllocatorBuilder::operator Allocator()
{
  return Allocator(m_strategy);
}


} // end of namespace umpire
