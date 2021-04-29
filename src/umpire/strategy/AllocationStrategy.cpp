//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/strategy/AllocationStrategy.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

AllocationStrategy::AllocationStrategy(const std::string& name, int id, AllocationStrategy* parent) noexcept
    : m_name{name}, m_id{id}, m_parent{parent}
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

void* AllocationStrategy::allocate_named(const std::string& UMPIRE_UNUSED_ARG(name), std::size_t UMPIRE_UNUSED_ARG(bytes))
{
  UMPIRE_ERROR("This allocation strategy does not support named allocations");

  //
  // The UMPIRE_ERROR macro above does not return.  It instead throws
  // an exception.  However, for some reason, nvcc throws a warning
  // "warning: missing return statement at end of non-void function"
  // even though the following line cannot be reached.  Adding this
  // fake return statement to work around the incorrect warning.
  //
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
  return nullptr;
#endif
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
