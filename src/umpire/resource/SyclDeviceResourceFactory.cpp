//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/SyclDeviceResourceFactory.hpp"

#include <memory>

#include "umpire/alloc/SyclMallocAllocator.hpp"
#include "umpire/resource/SyclDeviceMemoryResource.hpp"
#include "umpire/util/make_unique.hpp"

#if defined(UMPIRE_V1_DELEGATE_TO_V2)
#include "umpire/resource/sycl_device_memory.hpp"
#include "umpire/resource/v2_backed_resource.hpp"
#endif

namespace umpire {
namespace resource {

bool SyclDeviceResourceFactory::isValidMemoryResourceFor(const std::string& name) noexcept
{
  if ((name.find("CONST") == std::string::npos) && (name.find("DEVICE") != std::string::npos)) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<resource::MemoryResource> SyclDeviceResourceFactory::create(const std::string& name, int id)
{
  return create(name, id, getDefaultTraits());
}

std::unique_ptr<resource::MemoryResource> SyclDeviceResourceFactory::create(const std::string& name, int id,
                                                                            MemoryResourceTraits traits)
{
  auto sycl_asynchandler = [](sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception const& ex) {
        std::cout << "Caught asynchronous SYCL exception:" << std::endl
                  << ex.what() << ", OpenCL code: " << ex.code().value() << std::endl;
      }
    }
  };

  sycl::queue queue{sycl::gpu_selector_v};
  sycl::platform platform = queue.get_device().get_platform();

  int device_count = 0; // SYCL multi.device count
  auto const& devices = platform.get_devices();
  for (auto& device : devices) {
    if (device.is_gpu()) {
      if (device.get_info<sycl::info::device::partition_max_sub_devices>() > 0) {
        auto subDevicesDomainNuma =
            device.create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(
                sycl::info::partition_affinity_domain::numa);
        for (auto& subDev : subDevicesDomainNuma) {
          device_count++;
          if ((device_count - 1) == traits.id) {
            sycl::context syclctxt(subDev, sycl_asynchandler);
            traits.queue = new sycl::queue(syclctxt, subDev, sycl::property_list{sycl::property::queue::in_order{}});
          }
        }
      } else {
        device_count++;
        if ((device_count - 1) == traits.id) {
          sycl::context syclctxt(device, sycl_asynchandler);
          traits.queue = new sycl::queue(syclctxt, device, sycl::property_list{sycl::property::queue::in_order{}});
        }
      }
    }
  }

#if defined(UMPIRE_V1_DELEGATE_TO_V2)
  // Tracking=false: see the double-tracking discussion in
  // v2_backed_resource.hpp.
  //
  // Queue binding: v1's SyclDeviceMemoryResource passes `*traits.queue`
  // explicitly on every allocate()/deallocate() call, while v2's
  // sycl_device_memory binds a queue once at construction. Both models refer
  // to the same underlying SYCL queue (a reference-counted handle), so
  // constructing the v2 instance with a copy of the queue built above by the
  // factory (`*traits.queue`) is semantically equivalent.
  auto v2_memory =
      std::make_unique<resource::sycl_device_memory<resource::sycl_default_allocator, false>>(
          name + "_v2backed", *traits.queue);

  return util::make_unique<v2_backed_resource>(name, id, traits, Platform::sycl, std::move(v2_memory),
                                                [](Platform p) { return p == Platform::sycl; });
#else
  return util::make_unique<resource::SyclDeviceMemoryResource<alloc::SyclMallocAllocator>>(Platform::sycl, name, id,
                                                                                           traits);
#endif
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
