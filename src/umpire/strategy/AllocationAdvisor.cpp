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

#include "umpire/strategy/AllocationAdvisor.hpp"
#include "umpire/op/MemoryOperationRegistry.hpp"

#include "umpire/ResourceManager.hpp"

namespace umpire {
namespace strategy {

AllocationAdvisor::AllocationAdvisor(
    const std::string& name,
    int id,
    Allocator allocator,
    const std::string& advice_operation) :
  AllocationStrategy(name, id),
  m_current_size(0),
  m_highwatermark(0),
  m_allocator(allocator.getAllocationStrategy())
{
  auto& op_registry = op::MemoryOperationRegistry::getInstance();

  m_advice_operation = op_registry.find(
      advice_operation,
      m_allocator,
      m_allocator);
}

void* AllocationAdvisor::allocate(size_t bytes)
{

  void* ptr = m_allocator->allocate(bytes);
  auto alloc_record = new util::AllocationRecord{ptr, bytes, this->shared_from_this()};

  m_advice_operation->apply(
      ptr, 
      alloc_record, 
      0, 
      bytes);

  ResourceManager::getInstance().registerAllocation(ptr, alloc_record);

  m_current_size += bytes;
  if (m_current_size > m_highwatermark)
    m_highwatermark = m_current_size;

  return ptr;
}

void AllocationAdvisor::deallocate(void* ptr)
{
  m_allocator->deallocate(ptr);
  m_current_size -= ResourceManager::getInstance().getSize(ptr);

  ResourceManager::getInstance().deregisterAllocation(ptr);
}

long AllocationAdvisor::getCurrentSize()
{
  return m_current_size;
}

long AllocationAdvisor::getHighWatermark()
{
  return m_highwatermark;
}

Platform AllocationAdvisor::getPlatform()
{
  return m_allocator->getPlatform();
}

} // end of namespace strategy
} // end of namespace umpire
