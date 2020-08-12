//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
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
#include "umpire/op/CudaAdviseAccessedByOperation.hpp"
#include "umpire/op/CudaAdvisePreferredLocationOperation.hpp"
#include "umpire/op/CudaAdviseReadMostlyOperation.hpp"
#include "umpire/op/CudaAdviseUnsetAccessedByOperation.hpp"
#include "umpire/op/CudaAdviseUnsetPreferredLocationOperation.hpp"
#include "umpire/op/CudaAdviseUnsetReadMostlyOperation.hpp"
#include "umpire/op/CudaCopyFromOperation.hpp"
#include "umpire/op/CudaCopyOperation.hpp"
#include "umpire/op/CudaCopyToOperation.hpp"
#include "umpire/op/CudaMemPrefetchOperation.hpp"
#include "umpire/op/CudaMemsetOperation.hpp"
#endif

#if defined(UMPIRE_ENABLE_HIP)
#include "umpire/op/HipCopyFromOperation.hpp"
#include "umpire/op/HipCopyOperation.hpp"
#include "umpire/op/HipCopyToOperation.hpp"
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

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

MemoryOperationRegistry& MemoryOperationRegistry::getInstance() noexcept
{
  static MemoryOperationRegistry memory_operation_registry;
  return memory_operation_registry;
}

MemoryOperationRegistry::MemoryOperationRegistry() noexcept
{
  registerOperation("COPY", std::make_pair(Platform::host, Platform::host),
                    std::make_shared<HostCopyOperation>());

  registerOperation("MEMSET", std::make_pair(Platform::host, Platform::host),
                    std::make_shared<HostMemsetOperation>());

  registerOperation("REALLOCATE",
                    std::make_pair(Platform::host, Platform::host),
                    std::make_shared<HostReallocateOperation>());

#if defined(UMPIRE_ENABLE_NUMA)
  registerOperation("MOVE", std::make_pair(Platform::host, Platform::host),
                    std::make_shared<NumaMoveOperation>());

  // NOTE: We don't use CUDA calls in the move operation so no guard is needed
  registerOperation("MOVE", std::make_pair(Platform::host, Platform::cuda),
                    std::make_shared<NumaMoveOperation>());

  registerOperation("MOVE", std::make_pair(Platform::cuda, Platform::host),
                    std::make_shared<NumaMoveOperation>());
#endif

#if defined(UMPIRE_ENABLE_CUDA)
  registerOperation("COPY", std::make_pair(Platform::host, Platform::cuda),
                    std::make_shared<CudaCopyToOperation>());

  registerOperation("COPY", std::make_pair(Platform::cuda, Platform::host),
                    std::make_shared<CudaCopyFromOperation>());

  registerOperation("COPY", std::make_pair(Platform::cuda, Platform::cuda),
                    std::make_shared<CudaCopyOperation>());

  registerOperation("MEMSET", std::make_pair(Platform::cuda, Platform::cuda),
                    std::make_shared<CudaMemsetOperation>());

  registerOperation("REALLOCATE",
                    std::make_pair(Platform::cuda, Platform::cuda),
                    std::make_shared<GenericReallocateOperation>());

  registerOperation("ACCESSED_BY",
                    std::make_pair(Platform::cuda, Platform::cuda),
                    std::make_shared<CudaAdviseAccessedByOperation>());

  registerOperation("PREFERRED_LOCATION",
                    std::make_pair(Platform::cuda, Platform::cuda),
                    std::make_shared<CudaAdvisePreferredLocationOperation>());

  registerOperation("PREFERRED_LOCATION",
                    std::make_pair(Platform::host, Platform::host),
                    std::make_shared<CudaAdvisePreferredLocationOperation>());

  registerOperation("READ_MOSTLY",
                    std::make_pair(Platform::cuda, Platform::cuda),
                    std::make_shared<CudaAdviseReadMostlyOperation>());

  registerOperation("UNSET_ACCESSED_BY",
                    std::make_pair(Platform::cuda, Platform::cuda),
                    std::make_shared<CudaAdviseUnsetAccessedByOperation>());

  registerOperation(
      "UNSET_PREFERRED_LOCATION",
      std::make_pair(Platform::cuda, Platform::cuda),
      std::make_shared<CudaAdviseUnsetPreferredLocationOperation>());

  registerOperation(
      "UNSET_PREFERRED_LOCATION",
      std::make_pair(Platform::host, Platform::host),
      std::make_shared<CudaAdviseUnsetPreferredLocationOperation>());

  registerOperation("UNSET_READ_MOSTLY",
                    std::make_pair(Platform::cuda, Platform::cuda),
                    std::make_shared<CudaAdviseUnsetReadMostlyOperation>());

  registerOperation("PREFETCH", std::make_pair(Platform::cuda, Platform::cuda),
                    std::make_shared<CudaMemPrefetchOperation>());
#endif

#if defined(UMPIRE_ENABLE_HIP)
  registerOperation("COPY", std::make_pair(Platform::host, Platform::hip),
                    std::make_shared<HipCopyToOperation>());

  registerOperation("COPY", std::make_pair(Platform::hip, Platform::host),
                    std::make_shared<HipCopyFromOperation>());

  registerOperation("COPY", std::make_pair(Platform::hip, Platform::hip),
                    std::make_shared<HipCopyOperation>());

  registerOperation("MEMSET", std::make_pair(Platform::hip, Platform::hip),
                    std::make_shared<HipMemsetOperation>());

  registerOperation("REALLOCATE", std::make_pair(Platform::hip, Platform::hip),
                    std::make_shared<GenericReallocateOperation>());
#endif

#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
  registerOperation("COPY",
                    std::make_pair(Platform::host, Platform::omp_target),
                    std::make_shared<OpenMPTargetCopyOperation>());

  registerOperation("COPY",
                    std::make_pair(Platform::omp_target, Platform::host),
                    std::make_shared<OpenMPTargetCopyOperation>());

  registerOperation("COPY",
                    std::make_pair(Platform::omp_target, Platform::omp_target),
                    std::make_shared<OpenMPTargetCopyOperation>());

  registerOperation("MEMSET",
                    std::make_pair(Platform::omp_target, Platform::omp_target),
                    std::make_shared<OpenMPTargetMemsetOperation>());

  registerOperation("REALLOCATE",
                    std::make_pair(Platform::omp_target, Platform::omp_target),
                    std::make_shared<GenericReallocateOperation>());
#endif

#if defined(UMPIRE_ENABLE_SYCL)
  registerOperation("COPY", std::make_pair(Platform::host, Platform::sycl),
                    std::make_shared<SyclCopyToOperation>());

  registerOperation("COPY", std::make_pair(Platform::sycl, Platform::host),
                    std::make_shared<SyclCopyFromOperation>());

  registerOperation("COPY", std::make_pair(Platform::sycl, Platform::sycl),
                    std::make_shared<SyclCopyOperation>());

  registerOperation("MEMSET", std::make_pair(Platform::sycl, Platform::sycl),
                    std::make_shared<SyclMemsetOperation>());

  registerOperation("REALLOCATE",
                    std::make_pair(Platform::sycl, Platform::sycl),
                    std::make_shared<GenericReallocateOperation>());

  registerOperation("PREFETCH", std::make_pair(Platform::sycl, Platform::sycl),
                    std::make_shared<SyclMemPrefetchOperation>());
#endif
}

void MemoryOperationRegistry::registerOperation(
    const std::string& name, std::pair<Platform, Platform> platforms,
    std::shared_ptr<MemoryOperation>&& operation) noexcept
{
  auto operations = m_operators.find(name);

  if (operations == m_operators.end()) {
    operations =
        m_operators
            .insert(std::make_pair(
                name, std::unordered_map<std::pair<Platform, Platform>,
                                         std::shared_ptr<MemoryOperation>,
                                         pair_hash>()))
            .first;
  }

  operations->second.insert(std::make_pair(platforms, operation));
}

std::shared_ptr<umpire::op::MemoryOperation> MemoryOperationRegistry::find(
    const std::string& name, strategy::AllocationStrategy* src_allocator,
    strategy::AllocationStrategy* dst_allocator)
{
  auto platforms = std::make_pair(src_allocator->getPlatform(),
                                  dst_allocator->getPlatform());

  auto operations = m_operators.find(name);

  if (operations == m_operators.end()) {
    UMPIRE_ERROR("Cannot find operator " << name);
  }

  auto op = operations->second.find(platforms);

  if (op == operations->second.end()) {
    UMPIRE_ERROR("Cannot find operator" << name << " for platforms "
                                        << static_cast<int>(platforms.first)
                                        << ", "
                                        << static_cast<int>(platforms.second));
  }

  return op->second;
}

} // end of namespace op
} // end of namespace umpire
