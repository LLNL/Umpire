//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/GranularityController.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/op/MemoryOperationRegistry.hpp"

#if defined(UMPIRE_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#endif

namespace umpire {
namespace strategy {

GranularityController::GranularityController(const std::string& name, int id, Allocator allocator,
                                     Granularity granularity, int device_id)
    : AllocationStrategy{name, id, allocator.getAllocationStrategy(), "GranularityController"},
      m_allocator{allocator.getAllocationStrategy()},
      m_device{device_id},
      m_granularity{granularity}
{
}

void* GranularityController::allocate(std::size_t bytes)
{
  void* ptr = m_allocator->allocate_internal(bytes);

  return ptr;
}

void GranularityController::deallocate(void* ptr, std::size_t size)
{
  m_allocator->deallocate_internal(ptr, size);
}

Platform GranularityController::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

MemoryResourceTraits GranularityController::getTraits() const noexcept
{
  return m_allocator->getTraits();
}

} // end of namespace strategy
} // end of namespace umpire
