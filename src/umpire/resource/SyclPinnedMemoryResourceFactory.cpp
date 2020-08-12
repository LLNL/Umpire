//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/SyclPinnedMemoryResourceFactory.hpp"

#include "umpire/alloc/SyclPinnedAllocator.hpp"
#include "umpire/resource/SyclDeviceMemoryResource.hpp"
#include "umpire/util/make_unique.hpp"

namespace umpire {
namespace resource {

bool SyclPinnedMemoryResourceFactory::isValidMemoryResourceFor(
    const std::string& name) noexcept
{
  if (name.find("PINNED") != std::string::npos) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<resource::MemoryResource>
SyclPinnedMemoryResourceFactory::create(const std::string& name, int id)
{
  return create(name, id, getDefaultTraits());
}

std::unique_ptr<resource::MemoryResource>
SyclPinnedMemoryResourceFactory::create(const std::string& name, int id,
                                        MemoryResourceTraits traits)
{
  return util::make_unique<
      resource::SyclDeviceMemoryResource<alloc::SyclPinnedAllocator>>(
      Platform::sycl, name, id, traits);
}

MemoryResourceTraits SyclPinnedMemoryResourceFactory::getDefaultTraits()
{
  MemoryResourceTraits traits;

  cl::sycl::gpu_selector gpuSelect;
  cl::sycl::device sycl_device(gpuSelect);
  const std::string deviceName =
      sycl_device.get_info<cl::sycl::info::device::name>();
  if (sycl_device.is_gpu() &&
      (deviceName.find("Intel(R) Gen9 HD Graphics NEO") != std::string::npos)) {
    traits.size =
        0; // sycl_device.get_info<cl::sycl::info::device::global_mem_size>();
           // // in bytes
    traits.unified = false;

    traits.id = 0;

    traits.vendor = MemoryResourceTraits::vendor_type::INTEL;
    traits.kind = MemoryResourceTraits::memory_type::DDR;
    traits.used_for = MemoryResourceTraits::optimized_for::access;
  }

  return traits;
}

} // end of namespace resource
} // end of namespace umpire
