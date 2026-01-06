//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/MemoryOperationRegistry.hpp"

#include "umpire/config.hpp"
#include "umpire/op/GenericReallocateOperation.hpp"
#include "umpire/op/HostCopyOperation.hpp"
#include "umpire/op/HostMemsetOperation.hpp"
#include "umpire/op/HostReallocateOperation.hpp"

#if defined(UMPIRE_ENABLE_NUMA)
#include "umpire/op/NumaMoveOperation.hpp"
#endif

#if defined(UMPIRE_ENABLE_CUDA)
#include "umpire/op/CudaAdviseOperation.hpp"
#include "umpire/op/CudaCopyOperation.hpp"
#include "umpire/op/CudaMemPrefetchOperation.hpp"
#include "umpire/op/CudaMemsetOperation.hpp"
#endif

#if defined(UMPIRE_ENABLE_HIP)
#include "umpire/op/HipAdviseOperation.hpp"
#include "umpire/op/HipCopyOperation.hpp"
#include "umpire/op/HipMemsetOperation.hpp"
#endif

#if defined(UMPIRE_ENABLE_SYCL)
#include "umpire/op/SyclCopyFromOperation.hpp"
#include "umpire/op/SyclCopyOperation.hpp"
#include "umpire/op/SyclCopyToOperation.hpp"
#include "umpire/op/SyclMemPrefetchOperation.hpp"
#include "umpire/op/SyclMemsetOperation.hpp"
#endif

#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
#include <omp.h>

#include "umpire/op/OpenMPTargetCopyOperation.hpp"
#include "umpire/op/OpenMPTargetMemsetOperation.hpp"
#endif

#include "umpire/util/error.hpp"

namespace umpire {
namespace op {

MemoryOperationRegistry& MemoryOperationRegistry::getInstance() noexcept
{
  static MemoryOperationRegistry memory_operation_registry;
  return memory_operation_registry;
}

MemoryOperationRegistry::MemoryOperationRegistry() noexcept
{
  registerOperation("COPY", std::make_pair(Platform::host, Platform::host), std::make_shared<HostCopyOperation>());

  registerOperation("MEMSET", std::make_pair(Platform::host, Platform::host), std::make_shared<HostMemsetOperation>());

  registerOperation("REALLOCATE", std::make_pair(Platform::host, Platform::host),
                    std::make_shared<HostReallocateOperation>());

  registerOperation("REALLOCATE", std::make_pair(Platform::undefined, Platform::undefined),
                    std::make_shared<GenericReallocateOperation>());

#if defined(UMPIRE_ENABLE_NUMA)
  registerOperation("MOVE", std::make_pair(Platform::host, Platform::host), std::make_shared<NumaMoveOperation>());

  // NOTE: We don't use CUDA calls in the move operation so no guard is needed
  registerOperation("MOVE", std::make_pair(Platform::host, Platform::cuda), std::make_shared<NumaMoveOperation>());

  registerOperation("MOVE", std::make_pair(Platform::cuda, Platform::host), std::make_shared<NumaMoveOperation>());
#endif

#if defined(UMPIRE_ENABLE_CUDA)
  const std::tuple<std::string, cudaMemoryAdvise, umpire::Platform> cuda_advice_operations[] = {
      {"SET_READ_MOSTLY", cudaMemAdviseSetReadMostly, Platform::cuda},
      {"UNSET_READ_MOSTLY", cudaMemAdviseUnsetReadMostly, Platform::cuda},
      {"SET_PREFERRED_LOCATION", cudaMemAdviseSetPreferredLocation, Platform::cuda},
      {"UNSET_PREFERRED_LOCATION", cudaMemAdviseUnsetPreferredLocation, Platform::cuda},
      {"SET_ACCESSED_BY", cudaMemAdviseSetAccessedBy, Platform::cuda},
      {"UNSET_ACCESSED_BY", cudaMemAdviseUnsetAccessedBy, Platform::cuda},
      {"SET_PREFERRED_LOCATION", cudaMemAdviseSetPreferredLocation, Platform::host},
      {"UNSET_PREFERRED_LOCATION", cudaMemAdviseUnsetPreferredLocation, Platform::host},
      {"SET_ACCESSED_BY", cudaMemAdviseSetAccessedBy, Platform::host},
      {"UNSET_ACCESSED_BY", cudaMemAdviseUnsetAccessedBy, Platform::host}};

  const std::tuple<umpire::Platform, umpire::Platform, cudaMemcpyKind> cuda_copy_operations[] = {
      {Platform::host, Platform::cuda, cudaMemcpyHostToDevice},
      {Platform::cuda, Platform::host, cudaMemcpyDeviceToHost},
      {Platform::cuda, Platform::cuda, cudaMemcpyDeviceToDevice}};

  for (auto copy : cuda_copy_operations) {
    auto src_plat = std::get<0>(copy);
    auto dst_plat = std::get<1>(copy);
    auto kind = std::get<2>(copy);
    registerOperation("COPY", std::make_pair(src_plat, dst_plat), std::make_shared<CudaCopyOperation>(kind));
  }

  for (auto advice : cuda_advice_operations) {
    auto name = std::get<0>(advice);
    auto advice_enum = std::get<1>(advice);
    auto platform = std::get<2>(advice);
    registerOperation(name, std::make_pair(platform, platform), std::make_shared<CudaAdviseOperation>(advice_enum));
  }

  registerOperation("MEMSET", std::make_pair(Platform::cuda, Platform::cuda), std::make_shared<CudaMemsetOperation>());

  registerOperation("REALLOCATE", std::make_pair(Platform::cuda, Platform::cuda),
                    std::make_shared<GenericReallocateOperation>());

  registerOperation("PREFETCH", std::make_pair(Platform::cuda, Platform::cuda),
                    std::make_shared<CudaMemPrefetchOperation>());
#endif

#if defined(UMPIRE_ENABLE_HIP)
  const std::tuple<std::string, hipMemoryAdvise> hip_advice_operations[] = {
    {"SET_READ_MOSTLY", hipMemAdviseSetReadMostly},
    {"UNSET_READ_MOSTLY", hipMemAdviseUnsetReadMostly},
    {"SET_PREFERRED_LOCATION", hipMemAdviseSetPreferredLocation},
    {"UNSET_PREFERRED_LOCATION", hipMemAdviseUnsetPreferredLocation},
    {"SET_ACCESSED_BY", hipMemAdviseSetAccessedBy},
    {"UNSET_ACCESSED_BY", hipMemAdviseUnsetAccessedBy}
#if HIP_VERSION_MAJOR >= 5
    ,
    {"SET_COARSE_GRAIN", hipMemAdviseSetCoarseGrain},
    {"UNSET_COARSE_GRAIN", hipMemAdviseUnsetCoarseGrain}
#endif
  };

  const std::tuple<umpire::Platform, umpire::Platform, hipMemcpyKind> hip_copy_operations[] = {
      {Platform::host, Platform::hip, hipMemcpyHostToDevice},
      {Platform::hip, Platform::host, hipMemcpyDeviceToHost},
      {Platform::hip, Platform::hip, hipMemcpyDeviceToDevice}};

  for (auto copy : hip_copy_operations) {
    auto src_plat = std::get<0>(copy);
    auto dst_plat = std::get<1>(copy);
    auto kind = std::get<2>(copy);
    registerOperation("COPY", std::make_pair(src_plat, dst_plat), std::make_shared<HipCopyOperation>(kind));
  }

  for (auto advice : hip_advice_operations) {
    auto name = std::get<0>(advice);
    auto advice_enum = std::get<1>(advice);
    registerOperation(name, std::make_pair(Platform::hip, Platform::hip),
                      std::make_shared<HipAdviseOperation>(advice_enum));
  }

  registerOperation("MEMSET", std::make_pair(Platform::hip, Platform::hip), std::make_shared<HipMemsetOperation>());

  registerOperation("REALLOCATE", std::make_pair(Platform::hip, Platform::hip),
                    std::make_shared<GenericReallocateOperation>());
#endif

#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
  registerOperation("COPY", std::make_pair(Platform::host, Platform::omp_target),
                    std::make_shared<OpenMPTargetCopyOperation>());

  registerOperation("COPY", std::make_pair(Platform::omp_target, Platform::host),
                    std::make_shared<OpenMPTargetCopyOperation>());

  registerOperation("COPY", std::make_pair(Platform::omp_target, Platform::omp_target),
                    std::make_shared<OpenMPTargetCopyOperation>());

  registerOperation("MEMSET", std::make_pair(Platform::omp_target, Platform::omp_target),
                    std::make_shared<OpenMPTargetMemsetOperation>());

  registerOperation("REALLOCATE", std::make_pair(Platform::omp_target, Platform::omp_target),
                    std::make_shared<GenericReallocateOperation>());
#endif

#if defined(UMPIRE_ENABLE_SYCL)
  registerOperation("COPY", std::make_pair(Platform::host, Platform::sycl), std::make_shared<SyclCopyToOperation>());

  registerOperation("COPY", std::make_pair(Platform::sycl, Platform::host), std::make_shared<SyclCopyFromOperation>());

  registerOperation("COPY", std::make_pair(Platform::sycl, Platform::sycl), std::make_shared<SyclCopyOperation>());

  registerOperation("MEMSET", std::make_pair(Platform::sycl, Platform::sycl), std::make_shared<SyclMemsetOperation>());

  registerOperation("REALLOCATE", std::make_pair(Platform::sycl, Platform::sycl),
                    std::make_shared<GenericReallocateOperation>());

  registerOperation("PREFETCH", std::make_pair(Platform::sycl, Platform::sycl),
                    std::make_shared<SyclMemPrefetchOperation>());
#endif
}

void MemoryOperationRegistry::registerOperation(const std::string& name, std::pair<Platform, Platform> platforms,
                                                std::shared_ptr<MemoryOperation>&& operation) noexcept
{
  auto operations = m_operators.find(name);

  if (operations == m_operators.end()) {
    operations =
        m_operators
            .insert(std::make_pair(
                name, std::unordered_map<std::pair<Platform, Platform>, std::shared_ptr<MemoryOperation>, pair_hash>()))
            .first;
  }

  operations->second.insert(std::make_pair(platforms, operation));
}

std::shared_ptr<umpire::op::MemoryOperation> MemoryOperationRegistry::find(const std::string& name,
                                                                           strategy::AllocationStrategy* src_allocator,
                                                                           strategy::AllocationStrategy* dst_allocator)
{
  auto platforms = std::make_pair(src_allocator->getPlatform(), dst_allocator->getPlatform());

  return find(name, platforms);
}

std::shared_ptr<umpire::op::MemoryOperation> MemoryOperationRegistry::find(const std::string& name,
                                                                           std::pair<Platform, Platform> platforms)
{
  auto operations = m_operators.find(name);

  if (operations == m_operators.end()) {
    UMPIRE_ERROR(runtime_error, fmt::format("Cannot find operator \"{}\"", name));
  }

  auto op = operations->second.find(platforms);

  if (op == operations->second.end()) {
    UMPIRE_ERROR(runtime_error, fmt::format("Cannot find operator \"{}\" for platforms {}, {}", name,
                                            static_cast<int>(platforms.first), static_cast<int>(platforms.second)));
  }

  return op->second;
}

} // end of namespace op
} // end of namespace umpire
