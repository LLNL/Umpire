//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/Allocator.hpp"

#include "umpire/Replay.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {

Allocator::Allocator(strategy::AllocationStrategy* allocator) noexcept
    : strategy::mixins::Inspector{},
      strategy::mixins::AllocateNull{},
      m_allocator{allocator},
      m_tracking{allocator->isTracked()}
{
}

void Allocator::release()
{
  UMPIRE_REPLAY("\"event\": \"release\", \"payload\": { \"allocator_ref\": \""
                << m_allocator << "\" }");

  UMPIRE_LOG(Debug, "");

  m_allocator->release();
}

std::size_t Allocator::getSize(void* ptr) const
{
  UMPIRE_LOG(Debug, "(" << ptr << ")");
  return ResourceManager::getInstance().getSize(ptr);
}

std::size_t Allocator::getHighWatermark() const noexcept
{
 return m_allocator->getHighWatermark();
}

std::size_t Allocator::getCurrentSize() const noexcept
{
  return m_allocator->getCurrentSize();
}

std::size_t Allocator::getActualSize() const noexcept
{
  return m_allocator->getActualSize();
}

std::size_t Allocator::getAllocationCount() const noexcept
{
  return m_allocator->getAllocationCount();
}

const std::string& Allocator::getName() const noexcept
{
  return m_allocator->getName();
}

int Allocator::getId() const noexcept
{
  return m_allocator->getId();
}

strategy::AllocationStrategy* Allocator::getParent() const noexcept
{
  return m_allocator->getParent();
}

strategy::AllocationStrategy* Allocator::getAllocationStrategy() noexcept
{
  UMPIRE_LOG(Debug, "() returning " << m_allocator);
  return m_allocator;
}

Platform Allocator::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

bool Allocator::isTracked() const noexcept
{
  return m_allocator->isTracked();
}

const std::string& Allocator::getStrategyName() const noexcept
{
  return m_allocator->getStrategyName();
}

std::ostream& operator<<(std::ostream& os, const Allocator& allocator)
{
  os << *allocator.m_allocator;
  return os;
}

} // end of namespace umpire
