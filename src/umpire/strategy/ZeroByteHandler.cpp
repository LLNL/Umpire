//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/strategy/ZeroByteHandler.hpp"

#include "umpire/util/Macros.hpp"

#include "umpire/ResourceManager.hpp"

#include "umpire/strategy/FixedPool.hpp"

namespace umpire {
namespace strategy {

ZeroByteHandler::ZeroByteHandler(
  std::unique_ptr<AllocationStrategy>&& allocator) noexcept :
AllocationStrategy(allocator->getName(), allocator->getId()),
m_allocator(std::move(allocator))
{
}

void*
ZeroByteHandler::allocate(std::size_t bytes)
{
  if (0 == bytes) {
    UMPIRE_LOG(Debug, "Allocating 0 bytes for" << m_allocator->getName());
    auto& rm = ResourceManager::getInstance();
    return rm.getZeroByteAllocator()->allocate(1);
  } else {
    return m_allocator->allocate(bytes);
  }
}

void
ZeroByteHandler::deallocate(void* ptr)
{
  auto& rm = ResourceManager::getInstance();
  auto zero_pool = dynamic_cast<FixedPool*>(rm.getZeroByteAllocator());

  if (zero_pool->pointerIsFromPool(ptr)) {
    UMPIRE_LOG(Debug, "Deallocating 0 bytes for" << m_allocator->getName());
    zero_pool->deallocate(ptr);
  } else {
    m_allocator->deallocate(ptr);
  }
}

void
ZeroByteHandler::release()
{
  m_allocator->release();
}

std::size_t
ZeroByteHandler::getCurrentSize() const noexcept
{
  return m_allocator->getCurrentSize();
}

std::size_t
ZeroByteHandler::getHighWatermark() const noexcept
{
  return m_allocator->getHighWatermark();
}

std::size_t
ZeroByteHandler::getActualSize() const noexcept
{
  return m_allocator->getActualSize();
}

Platform
ZeroByteHandler::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

strategy::AllocationStrategy*
ZeroByteHandler::getAllocationStrategy()
{
  return m_allocator.get();
}

} // end of namespace umpire
} // end of namespace strategy
