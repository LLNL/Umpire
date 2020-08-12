//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/strategy/ZeroByteHandler.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

ZeroByteHandler::ZeroByteHandler(
    std::unique_ptr<AllocationStrategy>&& allocator) noexcept
    : AllocationStrategy(allocator->getName(), allocator->getId()),
      m_allocator(std::move(allocator)),
      m_zero_byte_pool(nullptr)
{
}

void* ZeroByteHandler::allocate(std::size_t bytes)
{
  if (0 == bytes) {
    UMPIRE_LOG(Debug, "Allocating 0 bytes for" << m_allocator->getName());
    if (!m_zero_byte_pool)
      m_zero_byte_pool = static_cast<FixedPool*>(
          ResourceManager::getInstance().getZeroByteAllocator());
    return m_zero_byte_pool->allocate(1);
  } else {
    return m_allocator->allocate(bytes);
  }
}

void ZeroByteHandler::deallocate(void* ptr)
{
  if (!m_zero_byte_pool)
    m_zero_byte_pool = static_cast<FixedPool*>(
        ResourceManager::getInstance().getZeroByteAllocator());

  if (m_zero_byte_pool->pointerIsFromPool(ptr)) {
    UMPIRE_LOG(Debug, "Deallocating 0 bytes for" << m_allocator->getName());
    m_zero_byte_pool->deallocate(ptr);
  } else {
    m_allocator->deallocate(ptr);
  }
}

void ZeroByteHandler::release()
{
  m_allocator->release();
}

std::size_t ZeroByteHandler::getCurrentSize() const noexcept
{
  return m_allocator->getCurrentSize();
}

std::size_t ZeroByteHandler::getHighWatermark() const noexcept
{
  return m_allocator->getHighWatermark();
}

std::size_t ZeroByteHandler::getActualSize() const noexcept
{
  return m_allocator->getActualSize();
}

Platform ZeroByteHandler::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

MemoryResourceTraits ZeroByteHandler::getTraits() const noexcept
{
  return m_allocator->getTraits();
}

strategy::AllocationStrategy* ZeroByteHandler::getAllocationStrategy()
{
  return m_allocator.get();
}

} // namespace strategy
} // namespace umpire
