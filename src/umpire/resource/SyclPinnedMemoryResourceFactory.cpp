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
      resource::SyclDeviceMemoryResource<alloc::SyclPinnedAllocator>>(
      Platform::sycl, name, id, traits);
}

MemoryResourceTraits SyclPinnedMemoryResourceFactory::getDefaultTraits()
{
  MemoryResourceTraits traits;

  traits.size = 0;
  traits.unified = false;

  traits.id = 0;

  traits.vendor = MemoryResourceTraits::vendor_type::intel;
  traits.kind = MemoryResourceTraits::memory_type::ddr;
  traits.used_for = MemoryResourceTraits::optimized_for::access;
  traits.resource = MemoryResourceTraits::resource_type::pinned;
  return traits;
}

} // end of namespace resource
} // end of namespace umpire
