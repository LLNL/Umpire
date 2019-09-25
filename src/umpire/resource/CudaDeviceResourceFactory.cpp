//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/CudaDeviceResourceFactory.hpp"

#include "umpire/resource/DefaultMemoryResource.hpp"
#include "umpire/alloc/CudaMallocAllocator.hpp"

#include <cuda_runtime_api.h>

#include "umpire/util/Macros.hpp"
#include "umpire/util/make_unique.hpp"

namespace umpire {
namespace resource {

bool
CudaDeviceResourceFactory::isValidMemoryResourceFor(const std::string& name)
  noexcept
{
  if (name.compare(handle()) == 0) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<resource::MemoryResource>
CudaDeviceResourceFactory::create(const std::string& UMPIRE_UNUSED_ARG(name), int id)
{
  MemoryResourceTraits traits;

  cudaDeviceProp properties;
  auto error = ::cudaGetDeviceProperties(&properties, 0);

  if (error != cudaSuccess) {
    UMPIRE_ERROR("cudaGetDeviceProperties failed with error: " << cudaGetErrorString(error));
  }

  traits.unified = false;
  traits.size = properties.totalGlobalMem;

  traits.vendor = MemoryResourceTraits::vendor_type::NVIDIA;
  traits.kind = MemoryResourceTraits::memory_type::GDDR;
  traits.used_for = MemoryResourceTraits::optimized_for::any;

  return util::make_unique<resource::DefaultMemoryResource<alloc::CudaMallocAllocator> >(Platform::cuda, "DEVICE", id, traits);
}

std::string CudaDeviceResourceFactory::handle() const noexcept
{
  return "DEVICE";
}

} // end of namespace resource
} // end of namespace umpire
