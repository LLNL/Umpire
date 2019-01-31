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

#if defined(UMPIRE_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#endif

namespace umpire {
namespace strategy {

AllocationAdvisor::AllocationAdvisor(
    const std::string& name,
    int id,
    Allocator allocator,
    const std::string& advice_operation) :
  AllocationAdvisor(
      name, id, allocator, advice_operation, allocator)
{
}

AllocationAdvisor::AllocationAdvisor(
    const std::string& name,
    int id,
    Allocator allocator,
    const std::string& advice_operation,
    Allocator accessing_allocator) :
  AllocationStrategy(name, id),
  m_allocator(allocator.getAllocationStrategy()),
  m_device(0)
{
  auto& op_registry = op::MemoryOperationRegistry::getInstance();

  m_advice_operation = op_registry.find(
      advice_operation,
      m_allocator,
      m_allocator);

#if defined(UMPIRE_ENABLE_CUDA)
  if (accessing_allocator.getPlatform() == Platform::cpu) {
    m_device = cudaCpuDeviceId;
  }
#else
  UMPIRE_USE_VAR(accessing_allocator);
#endif
}

void* AllocationAdvisor::allocate(size_t bytes)
{
  void* ptr = m_allocator->allocate(bytes);

  m_advice_operation->apply(
      ptr,
      nullptr,
      m_device,
      bytes);

  return ptr;
}

void AllocationAdvisor::deallocate(void* ptr)
{
  m_allocator->deallocate(ptr);

}

long AllocationAdvisor::getCurrentSize() const noexcept
{
  return 0;
}

long AllocationAdvisor::getHighWatermark() const noexcept
{
  return 0;
}

Platform AllocationAdvisor::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

} // end of namespace strategy
} // end of namespace umpire
