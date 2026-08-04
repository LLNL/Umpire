//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/CudaUnifiedMemoryResourceFactory.hpp"

#include <cuda_runtime_api.h>

#include <memory>

#include "umpire/alloc/CudaMallocManagedAllocator.hpp"
#include "umpire/resource/DefaultMemoryResource.hpp"
#include "umpire/util/error.hpp"
#include "umpire/util/make_unique.hpp"

#if defined(UMPIRE_V1_DELEGATE_TO_V2)
#include "umpire/resource/cuda_um_memory.hpp"
#include "umpire/resource/v2_backed_resource.hpp"
#endif

namespace umpire {
namespace resource {

bool CudaUnifiedMemoryResourceFactory::isValidMemoryResourceFor(const std::string& name) noexcept
{
  if (name.find("UM") != std::string::npos) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<resource::MemoryResource> CudaUnifiedMemoryResourceFactory::create(const std::string& name, int id)
{
  return create(name, id, getDefaultTraits());
}

std::unique_ptr<resource::MemoryResource> CudaUnifiedMemoryResourceFactory::create(const std::string& name, int id,
                                                                                   MemoryResourceTraits traits)
{
#if defined(UMPIRE_V1_DELEGATE_TO_V2)
  // Tracking=false: see the double-tracking discussion in
  // v2_backed_resource.hpp.
  auto v2_memory = std::make_unique<resource::cuda_um_memory<resource::cuda_um_allocator, false>>(name + "_v2backed");

  return util::make_unique<v2_backed_resource>(name, id, traits, Platform::cuda, std::move(v2_memory),
                                                [](Platform p) { return p == Platform::cuda || p == Platform::host; });
#else
  return util::make_unique<resource::DefaultMemoryResource<alloc::CudaMallocManagedAllocator>>(Platform::cuda, name, id,
                                                                                               traits);
#endif
}

MemoryResourceTraits CudaUnifiedMemoryResourceFactory::getDefaultTraits()
{
  MemoryResourceTraits traits;

  cudaDeviceProp properties;
  auto error = ::cudaGetDeviceProperties(&properties, 0);

  if (error != cudaSuccess) {
    UMPIRE_ERROR(runtime_error,
                 fmt::format("cudaGetDeviceProperties failed with error: {}", cudaGetErrorString(error)));
  }

  traits.unified = true;
  traits.size = properties.totalGlobalMem; // plus system size?

  traits.vendor = MemoryResourceTraits::vendor_type::nvidia;
  traits.kind = MemoryResourceTraits::memory_type::gddr;
  traits.used_for = MemoryResourceTraits::optimized_for::any;
  traits.resource = MemoryResourceTraits::resource_type::um;

  return traits;
}

} // end of namespace resource
} // end of namespace umpire
