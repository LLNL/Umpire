//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/config.hpp"

#include "umpire/resource/MemoryResourceRegistry.hpp"

#include "umpire/util/make_unique.hpp"

#include "umpire/resource/HostResourceFactory.hpp"
#include "umpire/resource/NullMemoryResourceFactory.hpp"

#if defined(UMPIRE_ENABLE_FILE_RESOURCE)
#include "umpire/resource/FileMemoryResourceFactory.hpp"
#endif

#if defined(UMPIRE_ENABLE_NUMA)
#include "umpire/strategy/NumaPolicy.hpp"
#endif

#if defined(UMPIRE_ENABLE_CUDA)
#include <cuda_runtime_api.h>

#include "umpire/resource/CudaDeviceResourceFactory.hpp"
#include "umpire/resource/CudaPinnedMemoryResourceFactory.hpp"
#include "umpire/resource/CudaUnifiedMemoryResourceFactory.hpp"
#if defined(UMPIRE_ENABLE_CONST)
#include "umpire/resource/CudaConstantMemoryResourceFactory.hpp"
#endif
#endif

#if defined(UMPIRE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "umpire/resource/HipDeviceResourceFactory.hpp"
#include "umpire/resource/HipPinnedMemoryResourceFactory.hpp"
#include "umpire/resource/HipUnifiedMemoryResourceFactory.hpp"
#if defined(UMPIRE_ENABLE_CONST)
#include "umpire/resource/HipConstantMemoryResourceFactory.hpp"
#endif
#endif

#if defined(UMPIRE_ENABLE_SYCL)
#include <CL/sycl.hpp>

#include "umpire/resource/SyclDeviceResourceFactory.hpp"
#include "umpire/resource/SyclPinnedMemoryResourceFactory.hpp"
#include "umpire/resource/SyclUnifiedMemoryResourceFactory.hpp"
#endif

#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
#include <omp.h>

#include "umpire/resource/OpenMPTargetMemoryResourceFactory.hpp"
#endif

#include "umpire/Replay.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace resource {

MemoryResourceRegistry& MemoryResourceRegistry::getInstance()
{
  static MemoryResourceRegistry resource_registry;

  return resource_registry;
}

const std::vector<std::string>&
MemoryResourceRegistry::getResourceNames() noexcept
{
  return m_resource_names;
}

MemoryResourceRegistry::MemoryResourceRegistry()
    : m_allocator_factories()
{
  registerMemoryResource(
      util::make_unique<resource::HostResourceFactory>());
  m_resource_names.push_back("HOST");

  registerMemoryResource(
      util::make_unique<resource::NullMemoryResourceFactory>());

#if defined(UMPIRE_ENABLE_FILE_RESOURCE)
  registerMemoryResource(
      util::make_unique<resource::FileMemoryResourceFactory>());
  m_resource_names.push_back("FILE");
#endif

#if defined(UMPIRE_ENABLE_CUDA)
  {
    int device_count{0};
    auto error = ::cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
      UMPIRE_ERROR("Umpire compiled with CUDA support but no GPUs detected!");
    }

    registerMemoryResource(
        util::make_unique<resource::CudaDeviceResourceFactory>());
    m_resource_names.push_back("DEVICE");

    for (int device = 0; device < device_count; ++device) {
      std::string name{"DEVICE::" + std::to_string(device)};
      m_resource_names.push_back(name);
    }

    registerMemoryResource(
        util::make_unique<resource::CudaUnifiedMemoryResourceFactory>());
    m_resource_names.push_back("UM");

    registerMemoryResource(
        util::make_unique<resource::CudaPinnedMemoryResourceFactory>());
    m_resource_names.push_back("PINNED");

#if defined(UMPIRE_ENABLE_CONST)
    registerMemoryResource(
        util::make_unique<resource::CudaConstantMemoryResourceFactory>());
    m_resource_names.push_back("DEVICE_CONST");
#endif
  }
#endif

#if defined(UMPIRE_ENABLE_HIP)
  {
    int device_count{0};
    auto error = ::hipGetDeviceCount(&device_count);
    if (error != hipSuccess) {
      UMPIRE_ERROR("Umpire compiled with HIP support but no GPUs detected!");
    }

    registerMemoryResource(
        util::make_unique<resource::HipDeviceResourceFactory>());
    m_resource_names.push_back("DEVICE");

    for (int device = 0; device < device_count; ++device) {
      std::string name{"DEVICE::" + std::to_string(device)};
      m_resource_names.push_back(name);
    }

    registerMemoryResource(
        util::make_unique<resource::HipUnifiedMemoryResourceFactory>());
    m_resource_names.push_back("UM");

    registerMemoryResource(
        util::make_unique<resource::HipPinnedMemoryResourceFactory>());
    m_resource_names.push_back("PINNED");

#if defined(UMPIRE_ENABLE_CONST)
    registerMemoryResource(
        util::make_unique<resource::HipConstantMemoryResourceFactory>());
    m_resource_names.push_back("DEVICE_CONST");
#endif
  }
#endif

#if defined(UMPIRE_ENABLE_SYCL)
  {
    int device_count{0};
    auto platforms = cl::sycl::platform::get_platforms();
    for (auto& platform : platforms) {
      auto devices = platform.get_devices();
      for (auto& device : devices) {
        const std::string deviceName =
            device.get_info<cl::sycl::info::device::name>();
        if (device.is_gpu() &&
            (deviceName.find("Intel(R) Gen9 HD Graphics NEO") !=
             std::string::npos))
          device_count++;
      }
    }

    if (device_count == 0) {
      UMPIRE_ERROR("Umpire compiled with SYCL support but no GPUs detected!");
    }

    registerMemoryResource(
        util::make_unique<resource::SyclDeviceResourceFactory>());
    m_resource_names.push_back("DEVICE");

    for (int device = 0; device < device_count; ++device) {
      std::string name{"DEVICE::" + std::to_string(device)};
      m_resource_names.push_back(name);
    }

    registerMemoryResource(
        util::make_unique<resource::SyclUnifiedMemoryResourceFactory>());
    m_resource_names.push_back("UM");

    registerMemoryResource(
        util::make_unique<resource::SyclPinnedMemoryResourceFactory>());
    m_resource_names.push_back("PINNED");
  }
#endif

#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
  int device_count{device_count = omp_get_num_devices()};
  for (int device = 0; device < device_count; ++device) {
    std::string name{"DEVICE::" + std::to_string(device)};
    m_resource_names.push_back(name);
  }

  registerMemoryResource(
      util::make_unique<resource::OpenMPTargetResourceFactory>());
  m_resource_names.push_back("DEVICE");
#endif
}

void MemoryResourceRegistry::registerMemoryResource(
    std::unique_ptr<MemoryResourceFactory>&& factory)
{
  m_allocator_factories.push_back(std::move(factory));
}

std::unique_ptr<resource::MemoryResource>
MemoryResourceRegistry::makeMemoryResource(const std::string& name, int id)
{
  for (auto const& allocator_factory : m_allocator_factories) {
    if (allocator_factory->isValidMemoryResourceFor(name)) {
      auto a = allocator_factory->create(name, id);
      return a;
    }
  }

  UMPIRE_ERROR("MemoryResource " << name << " not found");
}

std::unique_ptr<resource::MemoryResource>
MemoryResourceRegistry::makeMemoryResource(const std::string& name, int id,
                                           MemoryResourceTraits traits)
{
  for (auto const& allocator_factory : m_allocator_factories) {
    if (allocator_factory->isValidMemoryResourceFor(name)) {
      auto a = allocator_factory->create(name, id, traits);
      return a;
    }
  }

  UMPIRE_ERROR("MemoryResource " << name << " not found");
}

MemoryResourceTraits MemoryResourceRegistry::getDefaultTraitsForResource(
    const std::string& name)
{
  for (auto const& allocator_factory : m_allocator_factories) {
    if (allocator_factory->isValidMemoryResourceFor(name)) {
      return allocator_factory->getDefaultTraits();
    }
  }

  UMPIRE_ERROR("MemoryResource " << name << " not found");
}

} // end of namespace resource
} // end of namespace umpire
