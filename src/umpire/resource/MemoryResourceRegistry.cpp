//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/MemoryResourceRegistry.hpp"

#include "umpire/config.hpp"
#include "umpire/resource/HostResourceFactory.hpp"
#include "umpire/resource/NullMemoryResourceFactory.hpp"
#include "umpire/util/make_unique.hpp"

#if defined(UMPIRE_ENABLE_IPC_SHARED_MEMORY)
#include "umpire/resource/HostSharedMemoryResourceFactory.hpp"
#endif

#if defined(UMPIRE_ENABLE_MPI3_SHARED_MEMORY)
#include "umpire/resource/HostMpi3SharedMemoryResourceFactory.hpp"
#endif

#if defined(UMPIRE_ENABLE_DEVELOPER_BENCHMARKS)
#include "umpire/resource/NoOpResourceFactory.hpp"
#endif

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
#include "umpire/resource/SyclDeviceResourceFactory.hpp"
#include "umpire/resource/SyclPinnedMemoryResourceFactory.hpp"
#include "umpire/resource/SyclUnifiedMemoryResourceFactory.hpp"
#include "umpire/util/sycl_compat.hpp"
#endif

#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
#include <omp.h>

#include "umpire/resource/OpenMPTargetMemoryResourceFactory.hpp"
#endif

#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace resource {

MemoryResourceRegistry& MemoryResourceRegistry::getInstance()
{
  static MemoryResourceRegistry resource_registry;

  return resource_registry;
}

const std::vector<std::string>& MemoryResourceRegistry::getResourceNames() noexcept
{
  return m_resource_names;
}

MemoryResourceRegistry::MemoryResourceRegistry() : m_allocator_factories()
{
  registerMemoryResource(util::make_unique<resource::HostResourceFactory>());
  m_resource_names.push_back("HOST");

#if defined(UMPIRE_ENABLE_DEVELOPER_BENCHMARKS)
  registerMemoryResource(util::make_unique<resource::NoOpResourceFactory>());
  m_resource_names.push_back("NO_OP");
#endif

  registerMemoryResource(util::make_unique<resource::NullMemoryResourceFactory>());

#if defined(UMPIRE_ENABLE_IPC_SHARED_MEMORY)
  registerMemoryResource(util::make_unique<resource::HostSharedMemoryResourceFactory>());
  m_resource_names.push_back("SHARED::POSIX");
  if (std::string(UMPIRE_DEFAULT_SHARED_MEMORY_RESOURCE) == "POSIX") {
    m_resource_names.push_back("SHARED");
  }
#endif

#if defined(UMPIRE_ENABLE_MPI3_SHARED_MEMORY)
  registerMemoryResource(util::make_unique<resource::HostMpi3SharedMemoryResourceFactory>());
  m_resource_names.push_back("SHARED::MPI3");
  if (std::string(UMPIRE_DEFAULT_SHARED_MEMORY_RESOURCE) == "MPI3") {
    m_resource_names.push_back("SHARED");
  }
#endif

#if defined(UMPIRE_ENABLE_FILE_RESOURCE)
  registerMemoryResource(util::make_unique<resource::FileMemoryResourceFactory>());
  m_resource_names.push_back("FILE");
#endif

#if defined(UMPIRE_ENABLE_CUDA)
  {
    int device_count{0};
    auto error = ::cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
      UMPIRE_LOG(Warning, "Umpire compiled with CUDA support but no GPUs detected!");
    } else {
      registerMemoryResource(util::make_unique<resource::CudaDeviceResourceFactory>());
      m_resource_names.push_back("DEVICE");

      for (int device = 0; device < device_count; ++device) {
        std::string name{"DEVICE::" + std::to_string(device)};
        m_resource_names.push_back(name);
      }

      registerMemoryResource(util::make_unique<resource::CudaUnifiedMemoryResourceFactory>());
      m_resource_names.push_back("UM");

      registerMemoryResource(util::make_unique<resource::CudaPinnedMemoryResourceFactory>());
      m_resource_names.push_back("PINNED");

#if defined(UMPIRE_ENABLE_CONST)
      registerMemoryResource(util::make_unique<resource::CudaConstantMemoryResourceFactory>());
      m_resource_names.push_back("DEVICE_CONST");
#endif
    }
  }
#endif

#if defined(UMPIRE_ENABLE_HIP)
  {
    int coherence{0};
    auto error = hipDeviceGetAttribute(&coherence, hipDeviceAttributeFineGrainSupport, 0);
    const bool coherence_enabled{coherence == 1};
    int device_count{0};
    error = ::hipGetDeviceCount(&device_count);
    if (error != hipSuccess) {
      UMPIRE_ERROR(umpire::runtime_error,
                   fmt::format("Error! Can't get HIP device count: {}", hipGetErrorString(error)));
    } else {
      registerMemoryResource(util::make_unique<resource::HipDeviceResourceFactory>());
      m_resource_names.push_back("DEVICE");
#if defined(UMPIRE_ENABLE_HIP_COHERENCE_GRANULARITY)
      if (coherence_enabled) {
        m_resource_names.push_back("DEVICE::FINE");
        m_resource_names.push_back("DEVICE::COARSE");
      }
#endif

      for (int device = 0; device < device_count; ++device) {
        std::string name{"DEVICE::" + std::to_string(device)};
        m_resource_names.push_back(name);
#if defined(UMPIRE_ENABLE_HIP_COHERENCE_GRANULARITY)
        if (coherence_enabled) {
          m_resource_names.push_back(std::string{name + "::FINE"});
          m_resource_names.push_back(std::string{name + "::COARSE"});
        }
#endif
      }

      registerMemoryResource(util::make_unique<resource::HipUnifiedMemoryResourceFactory>());
      m_resource_names.push_back("UM");
#if defined(UMPIRE_ENABLE_HIP_COHERENCE_GRANULARITY)
      if (coherence_enabled) {
        m_resource_names.push_back("UM::FINE");
        m_resource_names.push_back("UM::COARSE");
      }
#endif

      registerMemoryResource(util::make_unique<resource::HipPinnedMemoryResourceFactory>());
      m_resource_names.push_back("PINNED");
#if defined(UMPIRE_ENABLE_HIP_COHERENCE_GRANULARITY)
      if (coherence_enabled) {
        m_resource_names.push_back("PINNED::FINE");
        m_resource_names.push_back("PINNED::COARSE");
      }
#endif

#if defined(UMPIRE_ENABLE_CONST)
      registerMemoryResource(util::make_unique<resource::HipConstantMemoryResourceFactory>());
      m_resource_names.push_back("DEVICE_CONST");
#endif
    }
    UMPIRE_USE_VAR(coherence_enabled);
  }
#endif

#if defined(UMPIRE_ENABLE_SYCL)
  {
    int device_count{0};
    auto platforms = sycl::platform::get_platforms();
    for (auto& platform : platforms) {
      auto devices = platform.get_devices();
      for (auto& device : devices) {
        if (device.is_gpu()) {
          if (device.get_info<sycl::info::device::partition_max_sub_devices>() > 0) {
            auto subDevicesDomainNuma =
                device.create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(
                    sycl::info::partition_affinity_domain::numa);
            device_count += subDevicesDomainNuma.size();
          } else
            device_count++;
        }
      }
    }

    if (device_count == 0) {
      UMPIRE_LOG(Warning, "Umpire compiled with SYCL support but no GPUs detected!");
    } else {
      registerMemoryResource(util::make_unique<resource::SyclDeviceResourceFactory>());
      m_resource_names.push_back("DEVICE");

      for (int device = 0; device < device_count; ++device) {
        std::string name{"DEVICE::" + std::to_string(device)};
        m_resource_names.push_back(name);
      }

      registerMemoryResource(util::make_unique<resource::SyclUnifiedMemoryResourceFactory>());
      m_resource_names.push_back("UM");

      registerMemoryResource(util::make_unique<resource::SyclPinnedMemoryResourceFactory>());
      m_resource_names.push_back("PINNED");
    }
  }
#endif

#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
  int device_count{device_count = omp_get_num_devices()};
  if (device_count >= 0) {
    for (int device = 0; device < device_count; ++device) {
      std::string name{"DEVICE::" + std::to_string(device)};
      m_resource_names.push_back(name);
    }

    registerMemoryResource(util::make_unique<resource::OpenMPTargetResourceFactory>());
    m_resource_names.push_back("DEVICE");
  } else {
    UMPIRE_LOG(Warning, "Umpire compiled with OpenMP Target support but no GPUs detected!");
  }

#endif
}

void MemoryResourceRegistry::registerMemoryResource(std::unique_ptr<MemoryResourceFactory>&& factory)
{
  m_allocator_factories.push_back(std::move(factory));
}

std::unique_ptr<resource::MemoryResource> MemoryResourceRegistry::makeMemoryResource(const std::string& name, int id)
{
  for (auto const& allocator_factory : m_allocator_factories) {
    if (allocator_factory->isValidMemoryResourceFor(name)) {
      auto a = allocator_factory->create(name, id);
      return a;
    }
  }

  UMPIRE_ERROR(runtime_error, fmt::format("MemoryResource \"{}\" not found", name));
}

std::unique_ptr<resource::MemoryResource> MemoryResourceRegistry::makeMemoryResource(const std::string& name, int id,
                                                                                     MemoryResourceTraits traits)
{
  for (auto const& allocator_factory : m_allocator_factories) {
    if (allocator_factory->isValidMemoryResourceFor(name)) {
      auto a = allocator_factory->create(name, id, traits);
      return a;
    }
  }

  UMPIRE_ERROR(runtime_error, fmt::format("MemoryResource \"{}\" not found", name));
}

MemoryResourceTraits MemoryResourceRegistry::getDefaultTraitsForResource(const std::string& name)
{
  for (auto const& allocator_factory : m_allocator_factories) {
    if (allocator_factory->isValidMemoryResourceFor(name)) {
      return allocator_factory->getDefaultTraits();
    }
  }

  UMPIRE_ERROR(runtime_error, fmt::format("MemoryResource \"{}\" not found", name));
}

} // end of namespace resource
} // end of namespace umpire
