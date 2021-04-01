//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/SyclUnifiedMemoryResourceFactory.hpp"

#include "umpire/alloc/SyclMallocManagedAllocator.hpp"
#include "umpire/resource/SyclDeviceMemoryResource.hpp"
#include "umpire/util/make_unique.hpp"

namespace umpire {
namespace resource {

bool SyclUnifiedMemoryResourceFactory::isValidMemoryResourceFor(
    const std::string& name) noexcept
{
  if (name.find("UM") != std::string::npos) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<resource::MemoryResource>
SyclUnifiedMemoryResourceFactory::create(const std::string& name, int id)
{
  return create(name, id, getDefaultTraits());
}

std::unique_ptr<resource::MemoryResource>
SyclUnifiedMemoryResourceFactory::create(const std::string& name, int id,
                                         MemoryResourceTraits traits)
{
  auto sycl_asynchandler = [] (sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
      try {
	std::rethrow_exception(e);
      } catch (sycl::exception const& ex) {
	std::cout << "Caught asynchronous SYCL exception:" << std::endl
	<< ex.what() << ", OpenCL code: " << ex.get_cl_code() << std::endl;
      }
    }
  };

  sycl::platform platform(sycl::gpu_selector{});

  int device_count = 0; // SYCL multi.device count
  auto const& devices = platform.get_devices();
  for (auto& device : devices) {
    if (device.is_gpu()) {
      if (device.get_info<sycl::info::device::partition_max_sub_devices>() > 0) {
	auto subDevicesDomainNuma = device.create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(sycl::info::partition_affinity_domain::numa);
	for (auto& subDev : subDevicesDomainNuma) {
	  device_count++;
	  if ((device_count-1) == traits.id) {
	    sycl::context syclctxt(subDev, sycl_asynchandler);
	    traits.queue = new sycl::queue(syclctxt, subDev, sycl::property_list{sycl::property::queue::in_order{}});
	  }
	}
      }
      else {
	device_count++;
	if ((device_count-1) == traits.id) {
	  sycl::context syclctxt(device, sycl_asynchandler);
	  traits.queue = new sycl::queue(syclctxt, device, sycl::property_list{sycl::property::queue::in_order{}});
	}
      }
    }
  }

  return util::make_unique<
      resource::SyclDeviceMemoryResource<alloc::SyclMallocManagedAllocator>>(
      Platform::sycl, name, id, traits);
}

MemoryResourceTraits SyclUnifiedMemoryResourceFactory::getDefaultTraits()
{
  MemoryResourceTraits traits;

  sycl::device syclDev(sycl::gpu_selector{});
  if (syclDev.is_gpu()) {
    if (syclDev.get_info<sycl::info::device::partition_max_sub_devices>() > 0) {
      auto subDevicesDomainNuma = syclDev.create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(
	sycl::info::partition_affinity_domain::numa);
      sycl::device subDev = subDevicesDomainNuma[0];

      traits.size =
        subDev.get_info<sycl::info::device::global_mem_size>(); // in bytes, plus system size?
      // abagusetty: bug with certain Intel devices for host_unified_memory flag returning false
      //subDev.get_info<sycl::info::device::host_unified_memory>();
    }
    else {
      traits.size =
        syclDev.get_info<sycl::info::device::global_mem_size>(); // in bytes
        //syclDev.get_info<sycl::info::device::host_unified_memory>();
    }

    traits.unified = true;
    traits.id = 0;

    traits.vendor = MemoryResourceTraits::vendor_type::intel;
    traits.kind = MemoryResourceTraits::memory_type::gddr;
    traits.used_for = MemoryResourceTraits::optimized_for::any;
    traits.resource = MemoryResourceTraits::resource_type::um;
  }

  return traits;
}

} // end of namespace resource
} // end of namespace umpire
