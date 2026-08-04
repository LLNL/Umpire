//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/CudaDeviceResourceFactory.hpp"

#include <cuda_runtime_api.h>

#include <memory>

#include "umpire/resource/CudaDeviceMemoryResource.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"
#include "umpire/util/make_unique.hpp"

#if defined(UMPIRE_V1_DELEGATE_TO_V2)
#include "umpire/resource/cuda_device_memory.hpp"
#include "umpire/resource/v2_backed_resource.hpp"
#endif

namespace umpire {
namespace resource {

bool CudaDeviceResourceFactory::isValidMemoryResourceFor(const std::string& name) noexcept
{
  if ((name.find("CONST") == std::string::npos) && (name.find("DEVICE") != std::string::npos)) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<resource::MemoryResource> CudaDeviceResourceFactory::create(const std::string& name, int id)
{
  return create(name, id, getDefaultTraits());
}

std::unique_ptr<resource::MemoryResource> CudaDeviceResourceFactory::create(const std::string& name, int id,
                                                                            MemoryResourceTraits traits)
{
#if defined(UMPIRE_V1_DELEGATE_TO_V2)
  // Delegate to the API v2 CUDA device resource. Tracking=false so v1's
  // ResourceManager::m_allocations remains the sole bookkeeping system (see
  // the double-tracking discussion in v2_backed_resource.hpp).
  //
  // Known behavioral nuance vs. native v1 CudaDeviceMemoryResource: v1
  // restores the previously-active CUDA device after each allocate()/
  // deallocate() call (see CudaDeviceMemoryResource.cpp), whereas v2's
  // cuda_default_allocator only calls cudaSetDevice(device_id) before the
  // operation and does not restore the prior device afterward. This can only
  // matter for multi-device programs that rely on the ambient
  // cudaGetDevice() being unchanged by an Umpire allocation call; it is
  // documented here since CUDA cannot be compile- or run-tested on this
  // host.
  auto v2_memory = std::make_unique<resource::cuda_device_memory<resource::cuda_default_allocator, false>>(
      name + "_v2backed", traits.id, resource::cuda_default_allocator(traits.id));

  return util::make_unique<v2_backed_resource>(name, id, traits, Platform::cuda, std::move(v2_memory),
                                                [](Platform p) { return p == Platform::cuda; });
#else
  return util::make_unique<resource::CudaDeviceMemoryResource>(Platform::cuda, name, id, traits);
#endif
}

MemoryResourceTraits CudaDeviceResourceFactory::getDefaultTraits()
{
  MemoryResourceTraits traits;

  cudaDeviceProp properties;
  auto error = ::cudaGetDeviceProperties(&properties, 0);

  if (error != cudaSuccess) {
    UMPIRE_ERROR(runtime_error,
                 fmt::format("cudaGetDeviceProperties failed with error: {}", cudaGetErrorString(error)));
  }

  traits.unified = false;
  traits.size = properties.totalGlobalMem;

  traits.vendor = MemoryResourceTraits::vendor_type::nvidia;
  traits.kind = MemoryResourceTraits::memory_type::gddr;
  traits.used_for = MemoryResourceTraits::optimized_for::any;
  traits.resource = MemoryResourceTraits::resource_type::device;

  traits.id = 0;

  return traits;
}

} // end of namespace resource
} // end of namespace umpire
