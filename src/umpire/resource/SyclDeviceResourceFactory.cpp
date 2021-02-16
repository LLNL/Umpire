//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/SyclDeviceResourceFactory.hpp"

#include "umpire/alloc/SyclMallocAllocator.hpp"
#include "umpire/resource/SyclDeviceMemoryResource.hpp"
#include "umpire/util/make_unique.hpp"

namespace umpire {
namespace resource {

bool SyclDeviceResourceFactory::isValidMemoryResourceFor(
    const std::string& name) noexcept
{
  if ((name.find("CONST") == std::string::npos) &&
      (name.find("DEVICE") != std::string::npos)) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<resource::MemoryResource> SyclDeviceResourceFactory::create(
    const std::string& name, int id)
{
  return create(name, id, getDefaultTraits());
}

std::unique_ptr<resource::MemoryResource> SyclDeviceResourceFactory::create(
    const std::string& name, int id, MemoryResourceTraits traits)
{
  auto platforms = cl::sycl::platform::get_platforms();
  cl::sycl::device d;

  for (auto& platform : platforms) {
    auto devices = platform.get_devices();
    for (auto& device : devices) {
      int device_count = 0; // SYCL multi.device count
      const std::string deviceName =
          device.get_info<cl::sycl::info::device::name>();
      if (device.is_gpu()) {
        if (device_count == traits.id) {
          d = device;
        }
        device_count++;
      }
    }
  }

  cl::sycl::queue sycl_queue(d);
  traits.queue = sycl_queue;
  return util::make_unique<
      resource::SyclDeviceMemoryResource<alloc::SyclMallocAllocator>>(
      Platform::sycl, name, id, traits);
}

MemoryResourceTraits SyclDeviceResourceFactory::getDefaultTraits()
{
  MemoryResourceTraits traits;

  traits.unified = false;
  traits.id = 0;
  traits.vendor = MemoryResourceTraits::vendor_type::intel;
  traits.kind = MemoryResourceTraits::memory_type::gddr;
  traits.used_for = MemoryResourceTraits::optimized_for::any;
  traits.resource = MemoryResourceTraits::resource_type::device;

  return traits;
}

} // end of namespace resource
} // end of namespace umpire
