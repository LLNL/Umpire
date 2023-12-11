//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/strategy/StreamAwareAllocationStrategy.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

StreamAwareAllocationStrategy::StreamAwareAllocationStrategy(const std::string& name, int id, AllocationStrategy* parent,
                                       const std::string& strategy_name) noexcept
    : m_name{name}, m_strategy_name{strategy_name}, m_id{id}, m_parent{parent}
{
}

void* StreamAwareAllocationStrategy::allocate_internal(std::size_t bytes)
{
  m_current_size += bytes;
  m_allocation_count++;

  if (m_current_size > m_high_watermark) {
    m_high_watermark = m_current_size;
  }

  return allocate(bytes);
}

void* StreamAwareAllocationStrategy::allocate_named(const std::string& UMPIRE_UNUSED_ARG(name), std::size_t bytes)
{
  return allocate(bytes);
}

//do something with stream?
void StreamAwareAllocationStrategy::allocate(void* stream, std::size_t bytes)
{
  allocate(bytes);
}

void StreamAwareAllocationStrategy::deallocate_internal(void* ptr, std::size_t size)
{
  m_current_size -= size;
  m_allocation_count--;

  deallocate(ptr, size);
}

//do something with stream?
void StreamAwareAllocationStrategy::deallocate(void* stream, void* ptr, std::size_t size)
{
  return deallocate(ptr, size);
}

const std::string& StreamAwareAllocationStrategy::getName() noexcept
{
  return m_name;
}

const std::string& StreamAwareAllocationStrategy::getStrategyName() const noexcept
{
  return m_strategy_name;
}

void StreamAwareAllocationStrategy::release()
{
  UMPIRE_LOG(Info, "StreamAwareAllocationStrategy::release is a no-op");
}

int StreamAwareAllocationStrategy::getId() noexcept
{
  return m_id;
}

std::size_t StreamAwareAllocationStrategy::getCurrentSize() const noexcept
{
  return m_current_size;
}

std::size_t StreamAwareAllocationStrategy::getHighWatermark() const noexcept
{
  return m_high_watermark;
}

std::size_t StreamAwareAllocationStrategy::getAllocationCount() const noexcept
{
  return m_allocation_count;
}

std::size_t StreamAwareAllocationStrategy::getActualSize() const noexcept
{
  return getCurrentSize();
}

MemoryResourceTraits StreamAwareAllocationStrategy::getTraits() const noexcept
{
  UMPIRE_LOG(Error, "StreamAwareAllocationStrategy::getTraits() not implemented");

  return MemoryResourceTraits{};
}

AllocationStrategy* AllocationStrategy::getParent() const noexcept
{
  return m_parent;
}

bool StreamAwareAllocationStrategy::tracksMemoryUse() const noexcept
{
  return false;
}

void StreamAwareAllocationStrategy::setTracking(bool tracking) noexcept
{
  m_tracked = tracking;
}

bool StreamAwareAllocationStrategy::isTracked() const noexcept
{
  return m_tracked;
}

std::ostream& operator<<(std::ostream& os, const StreamAwareAllocationStrategy& strategy)
{
  os << "[" << strategy.m_name << "," << strategy.m_id << "]";
  return os;
}

} // end of namespace strategy
} // end of namespace umpire
