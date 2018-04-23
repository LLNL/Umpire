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
#include "umpire/Allocator.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"

#include "umpire/util/StatisticsDatabase.hpp"
#include "umpire/util/Statistic.hpp"

namespace umpire {

Allocator::Allocator(std::shared_ptr<strategy::AllocationStrategy>& allocator):
  m_allocator(allocator)
{
}

void*
Allocator::allocate(size_t bytes)
{
  UMPIRE_LOG(Debug, "(" << bytes << ")");
  void* ret = m_allocator->allocate(bytes);

  util::Statistic::AllocationStatistic stat = {ret, bytes, "allocate"};
  util::StatisticsDatabase::getDatabase()->getStatistic(getName(), util::Statistic::ALLOC_STAT)->recordAllocationStatistic(stat);

  return ret;
}

void
Allocator::deallocate(void* ptr)
{
  UMPIRE_ASSERT("Deallocate called with nullptr" && ptr);
  UMPIRE_LOG(Debug, "(" << ptr << ")");

  util::Statistic::AllocationStatistic stat = {ptr, 0x0, "deallocate"};
  util::StatisticsDatabase::getDatabase()->getStatistic(getName(), util::Statistic::ALLOC_STAT)->recordAllocationStatistic(stat);

  m_allocator->deallocate(ptr);
}

size_t
Allocator::getSize(void* ptr)
{
  UMPIRE_LOG(Debug, "(" << ptr << ")");
  return ResourceManager::getInstance().getSize(ptr);
}

size_t
Allocator::getHighWatermark()
{
  return m_allocator->getHighWatermark();
}

size_t
Allocator::getCurrentSize()
{
  return m_allocator->getCurrentSize();
}

std::string
Allocator::getName()
{
  return m_allocator->getName();
}

int
Allocator::getId()
{
  return m_allocator->getId();
}

std::shared_ptr<strategy::AllocationStrategy>
Allocator::getAllocationStrategy()
{
  UMPIRE_LOG(Debug, "() returning " << m_allocator);
  return m_allocator;
}

} // end of namespace umpire
