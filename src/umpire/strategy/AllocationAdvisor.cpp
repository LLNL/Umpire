//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
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
    const std::string& advice_operation,
    int device_id) :
  AllocationAdvisor(
      name, id, allocator, advice_operation, allocator, device_id)
{
}

AllocationAdvisor::AllocationAdvisor(
    const std::string& name,
    int id,
    Allocator allocator,
    const std::string& advice_operation,
    Allocator accessing_allocator,
    int device_id) :
  AllocationStrategy(name, id),
  m_size_map{}
  m_allocator{allocator.getAllocationStrategy()},
  m_device{device_id}
{
  auto& op_registry = op::MemoryOperationRegistry::getInstance();

  m_set_advice_operation = op_registry.find(
      advice_operation,
      m_allocator,
      m_allocator);

  const std::string unset_operation = "UNSET_" + advice_operation;
  m_unset_advice_operation = op_registry.find(
      unset_operation,
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

void* AllocationAdvisor::allocate(std::size_t bytes)
{
  void* ptr = m_allocator->allocate(bytes);

  m_set_advice_operation->apply(
      ptr,
      nullptr,
      m_device,
      bytes);

  m_size_map[ptr] = bytes;

  return ptr;
}

void AllocationAdvisor::deallocate(void* ptr)
{
  auto bytes = m_size_map[ptr];

  m_unset_advice_operation->apply(
      ptr,
      nullptr,
      m_device,
      bytes);

  m_allocator->deallocate(ptr);

}

std::size_t AllocationAdvisor::getCurrentSize() const noexcept
{
  return 0;
}

std::size_t AllocationAdvisor::getHighWatermark() const noexcept
{
  return 0;
}

Platform AllocationAdvisor::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

} // end of namespace strategy
} // end of namespace umpire
