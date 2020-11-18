//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/Memory.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {

Memory::Memory(const std::string& name, int id, Memory* parent) noexcept
    : m_name(name), m_id(id), m_parent(parent) 
{
}

const std::string& Memory::getName() noexcept
{
  return m_name;
}

void Memory::release()
{
  UMPIRE_LOG(Info, "Memory::release is a no-op");
}

int Memory::getId() noexcept
{
  return m_id;
}

std::size_t Memory::getCurrentSize() const noexcept
{
  return 0;
}

std::size_t Memory::getHighWatermark() const noexcept
{
  return 0;
}

std::size_t Memory::getAllocationCount() const noexcept
{
  return 0;
}

std::size_t Memory::getActualSize() const noexcept
{
  return getCurrentSize();
}

MemoryResourceTraits Memory::getTraits() const noexcept
{
  UMPIRE_LOG(Error, "Memory::getTraits() not implemented");

  return MemoryResourceTraits{};
}

Memory* Memory::getParent() const noexcept
{
  return m_parent;
}

std::ostream& operator<<(std::ostream& os, const Memory& memory)
{
  os << "[" << memory.m_name << "," << memory.m_id << "]";
  return os;
}

} // end of namespace umpire
