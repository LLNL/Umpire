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

AllocationStrategy::AllocationStrategy(const std::string& name, int id) noexcept
    : m_name(name), m_id(id)
{
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
  return 0;
}

std::size_t AllocationStrategy::getHighWatermark() const noexcept
{
  return 0;
}

std::size_t AllocationStrategy::getAllocationCount() const noexcept
{
  return 0;
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

std::ostream& operator<<(std::ostream& os, const AllocationStrategy& strategy)
{
  os << "[" << strategy.m_name << "," << strategy.m_id << "]";
  return os;
}

} // end of namespace strategy
} // end of namespace umpire
