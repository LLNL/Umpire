//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/strategy/AllocationStrategy.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

AllocationStrategy::AllocationStrategy(const std::string& name, int id, AllocationStrategy* parent,
                                       const std::string& strategy_name) noexcept
    : m_name{name}, m_strategy_name{strategy_name}, m_id{id}, m_parent{parent}
{
}

void* AllocationStrategy::allocate_internal(std::size_t bytes)
{
  m_current_size += bytes;
  m_allocation_count++;

  if (m_current_size > m_high_watermark) {
    m_high_watermark = m_current_size;
  }

  return allocate(bytes);
}

void* AllocationStrategy::allocate_named(const std::string& UMPIRE_UNUSED_ARG(name), std::size_t bytes)
{
  return allocate(bytes);
}

void* AllocationStrategy::allocate_resource(std::size_t bytes, camp::resources::Resource UMPIRE_UNUSED_ARG(r))
{
  return allocate(bytes);
}

void AllocationStrategy::deallocate_resource(void* ptr, camp::resources::Resource UMPIRE_UNUSED_ARG(r),
                                             std::size_t size)
{
  deallocate(ptr, size);
}

void* AllocationStrategy::allocate_named_internal(const std::string& name, std::size_t bytes)
{
  m_current_size += bytes;
  m_allocation_count++;

  if (m_current_size > m_high_watermark) {
    m_high_watermark = m_current_size;
  }

  return allocate_named(name, bytes);
}

void AllocationStrategy::deallocate_internal(void* ptr, std::size_t size)
{
  m_current_size -= size;
  m_allocation_count--;

  deallocate(ptr, size);
}

const std::string& AllocationStrategy::getName() noexcept
{
  return m_name;
}

const std::string& AllocationStrategy::getStrategyName() const noexcept
{
  return m_strategy_name;
}

void AllocationStrategy::release()
{
  UMPIRE_LOG(Info, "AllocationStrategy::release is a no-op");
}

int AllocationStrategy::getId() noexcept
{
  return m_id;
}

std::size_t AllocationStrategy::getCurrentSize() const noexcept
{
  return m_current_size;
}

std::size_t AllocationStrategy::getHighWatermark() const noexcept
{
  return m_high_watermark;
}

std::size_t AllocationStrategy::getAllocationCount() const noexcept
{
  return m_allocation_count;
}

std::size_t AllocationStrategy::getActualSize() const noexcept
{
  return getCurrentSize();
}

MemoryResourceTraits AllocationStrategy::getTraits() const noexcept
{
  UMPIRE_LOG(Error, "AllocationStrategy::getTraits() not implemented");

  return MemoryResourceTraits{};
}

AllocationStrategy* AllocationStrategy::getParent() const noexcept
{
  return m_parent;
}

bool AllocationStrategy::tracksMemoryUse() const noexcept
{
  return false;
}

void AllocationStrategy::setTracking(bool tracking) noexcept
{
  m_tracked = tracking;
}

bool AllocationStrategy::isTracked() const noexcept
{
  return m_tracked;
}

std::ostream& operator<<(std::ostream& os, const AllocationStrategy& strategy)
{
  os << "[" << strategy.m_name << "," << strategy.m_id << "]";
  return os;
}

} // end of namespace strategy
} // end of namespace umpire
