//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/SyclUnifiedMemoryResourceFactory.hpp"

#include "umpire/resource/DefaultMemoryResource.hpp"

#include "umpire/alloc/SyclMallocManagedAllocator.hpp"

#include "umpire/util/make_unique.hpp"

#include <CL/sycl.hpp>


namespace umpire {
namespace resource {

bool
SyclUnifiedMemoryResourceFactory::isValidMemoryResourceFor(const std::string& name)
  noexcept
{
  // can easily do it with host_unified_memory<>()
  if (name.find("UM") != std::string::npos) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<resource::MemoryResource>
CudaUnifiedMemoryResourceFactory::create(const std::string& name, int id)
{
  return create(name, id, getDefaultTraits());
}

std::unique_ptr<resource::MemoryResource>
CudaUnifiedMemoryResourceFactory::create(const std::string& name, int id, MemoryResourceTraits traits)
{
  return
    util::make_unique<resource::DefaultMemoryResource<alloc::SyclMallocManagedAllocator>>(Platform::sycl, name, id, traits);
}

MemoryResourceTraits
SyclUnifiedMemoryResourceFactory::getDefaultTraits()
{
  MemoryResourceTraits traits;
  
  auto platforms = platform::get_platforms();
  for (auto &platform : platforms) {
    auto devices = platform.get_devices();
    for (auto &device : devices) {
      const std::string deviceName = device.get_info<info::device::name>();
      if (device.is_gpu() && (deviceName.find("Intel") != std::string::npos)) {
        traits.unified = dev.get_info<info::device::host_unified_memory>();        
        traits.size = dev.get_info<info::device::global_mem_size>(); // in bytes
        traits.id = dev.get(); // equivalent to cl_device_id
        traits.vendor = MemoryResourceTraits::vendor_type::INTEL;

        break;
      }
    }
  }

  traits.kind = MemoryResourceTraits::memory_type::GDDR; // todo: check this?
  traits.used_for = MemoryResourceTraits::optimized_for::any; 

  return traits;
}

} // end of namespace resource
} // end of namespace umpire
