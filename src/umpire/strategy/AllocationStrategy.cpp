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
#include "umpire/strategy/AllocationStrategy.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

AllocationStrategy::AllocationStrategy(const std::string& name, int id) noexcept :
  m_name(name),
  m_id(id)
{
}

void
AllocationStrategy::release()
{
  UMPIRE_LOG(Info, "AllocationStrategy::release in a no-op");
}

std::string
AllocationStrategy::getName() noexcept
{
  return m_name;
}

int
AllocationStrategy::getId() noexcept
{
  return m_id;
}

long
AllocationStrategy::getActualSize() const noexcept
{
  return getCurrentSize();
}

} // end of namespace strategy
} // end of namespace umpire
