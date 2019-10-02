//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
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

void
Allocator::release()
{
  UMPIRE_REPLAY("\"event\": \"release\", \"payload\": { \"allocator_ref\": \"" <<  m_allocator << "\" }");

  UMPIRE_LOG(Debug, "");

  m_allocator->release();
}

std::size_t
Allocator::getSize(void* ptr) const
{
  UMPIRE_LOG(Debug, "(" << ptr << ")");
  return ResourceManager::getInstance().getSize(ptr);
}

std::size_t
Allocator::getHighWatermark() const noexcept
{
  return m_allocator->getHighWatermark();
}

std::size_t
Allocator::getCurrentSize() const noexcept
{
  return m_allocator->getCurrentSize();
}

std::size_t
Allocator::getActualSize() const noexcept
{
  return (m_allocator->getActualSize() > 0) ? 
    m_allocator->getActualSize() : m_allocator->getCurrentSize();
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
