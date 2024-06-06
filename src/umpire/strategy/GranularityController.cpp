//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/GranularityController.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/alloc/HipAllocator.hpp"
#include "umpire/op/MemoryOperationRegistry.hpp"
#include "umpire/resource/HipMemoryResource.hpp"

#if defined(UMPIRE_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#endif

namespace umpire {
namespace strategy {

GranularityController::GranularityController(const std::string& name, int id, Allocator allocator,
                                             Granularity granularity)
    : AllocationStrategy{name, id, allocator.getAllocationStrategy(), "GranularityController"},
      m_allocator{allocator.getAllocationStrategy()},
      m_granularity{granularity}
{
  umpire::resource::HipMemoryResource* strat{dynamic_cast<umpire::resource::HipMemoryResource*>(m_allocator)};

  if (strat == nullptr) {
    UMPIRE_ERROR(runtime_error, fmt::format("Cannot place Granularity Controller atop non-HIP device"));
  }
}

void* GranularityController::allocate(std::size_t bytes)
{
  umpire::resource::HipMemoryResource* strat{dynamic_cast<umpire::resource::HipMemoryResource*>(m_allocator)};

  Granularity old_granularity{strat->set_granularity(m_granularity)};

  void* ptr{m_allocator->allocate_internal(bytes)};

  strat->set_granularity(old_granularity);

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
