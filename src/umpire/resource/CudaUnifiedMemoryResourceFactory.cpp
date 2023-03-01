//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/CudaUnifiedMemoryResourceFactory.hpp"

#include <cuda_runtime_api.h>

#include "umpire/alloc/CudaMallocManagedAllocator.hpp"
#include "umpire/resource/DefaultMemoryResource.hpp"
#include "umpire/util/error.hpp"
#include "umpire/util/make_unique.hpp"

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
  return util::make_unique<resource::DefaultMemoryResource<alloc::CudaMallocManagedAllocator>>(Platform::cuda, name, id,
                                                                                               traits);
}

MemoryResourceTraits CudaUnifiedMemoryResourceFactory::getDefaultTraits()
{
  MemoryResourceTraits traits;

  cudaDeviceProp properties;
  auto error = ::cudaGetDeviceProperties(&properties, 0);

  if (error != cudaSuccess) {
    UMPIRE_ERROR(runtime_error,
                 umpire::fmt::format("cudaGetDeviceProperties failed with error: {}", cudaGetErrorString(error)));
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
