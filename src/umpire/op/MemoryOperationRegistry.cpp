//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////
#include "umpire/config.hpp"

#include "umpire/op/MemoryOperationRegistry.hpp"

#include "umpire/op/HostCopyOperation.hpp"
#include "umpire/op/HostMemsetOperation.hpp"
#include "umpire/op/HostReallocateOperation.hpp"

#include "umpire/op/GenericReallocateOperation.hpp"

#if defined(UMPIRE_ENABLE_CUDA)
#include "umpire/op/CudaCopyFromOperation.hpp"
#include "umpire/op/CudaCopyToOperation.hpp"
#include "umpire/op/CudaCopyOperation.hpp"

#include "umpire/op/CudaMemsetOperation.hpp"

#include "umpire/op/CudaAdviseAccessedByOperation.hpp"
#include "umpire/op/CudaAdvisePreferredLocationOperation.hpp"
#include "umpire/op/CudaAdviseReadMostlyOperation.hpp"
#endif

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

MemoryOperationRegistry*
MemoryOperationRegistry::s_memory_operation_registry_instance = nullptr;

MemoryOperationRegistry&
MemoryOperationRegistry::getInstance() noexcept
{
  if (!s_memory_operation_registry_instance) {
    s_memory_operation_registry_instance = new MemoryOperationRegistry();
    UMPIRE_LOG(Debug, "() Created MemoryOperationRegistry at " << s_memory_operation_registry_instance);
  }

  return *s_memory_operation_registry_instance;
}

MemoryOperationRegistry::MemoryOperationRegistry() noexcept
{
  registerOperation(
      "COPY",
      std::make_pair(Platform::cpu, Platform::cpu),
      std::make_shared<HostCopyOperation>());

  registerOperation(
      "MEMSET",
      std::make_pair(Platform::cpu, Platform::cpu),
      std::make_shared<HostMemsetOperation>());

  registerOperation(
      "REALLOCATE",
      std::make_pair(Platform::cpu, Platform::cpu),
      std::make_shared<HostReallocateOperation>());

#if defined(UMPIRE_ENABLE_CUDA)
  registerOperation(
      "COPY",
      std::make_pair(Platform::cpu, Platform::cuda),
      std::make_shared<CudaCopyToOperation>());

  registerOperation(
      "COPY",
      std::make_pair(Platform::cuda, Platform::cpu),
      std::make_shared<CudaCopyFromOperation>());

  registerOperation(
      "COPY",
      std::make_pair(Platform::cuda, Platform::cuda),
      std::make_shared<CudaCopyOperation>());

  registerOperation(
      "MEMSET",
      std::make_pair(Platform::cuda, Platform::cuda),
      std::make_shared<CudaMemsetOperation>());

  registerOperation(
      "REALLOCATE",
      std::make_pair(Platform::cuda, Platform::cuda),
      std::make_shared<GenericReallocateOperation>());

  registerOperation(
      "ACCESSED_BY",
      std::make_pair(Platform::cuda, Platform::cuda),
      std::make_shared<CudaAdviseAccessedByOperation>());

  registerOperation(
      "PREFERRED_LOCATION",
      std::make_pair(Platform::cuda, Platform::cuda),
      std::make_shared<CudaAdvisePreferredLocationOperation>());

  registerOperation(
      "PREFERRED_LOCATION",
      std::make_pair(Platform::cpu, Platform::cpu),
      std::make_shared<CudaAdvisePreferredLocationOperation>());

  registerOperation(
      "READ_MOSTLY",
      std::make_pair(Platform::cuda, Platform::cuda),
      std::make_shared<CudaAdviseReadMostlyOperation>());

#endif
}

void
MemoryOperationRegistry::registerOperation(
    const std::string& name,
    std::pair<Platform, Platform> platforms,
    std::shared_ptr<MemoryOperation>&& operation) noexcept
{
  auto operations = m_operators.find(name);

  if (operations == m_operators.end()) {
    operations = m_operators.insert(
        std::make_pair(name,
          std::unordered_map<std::pair<Platform, Platform>,
          std::shared_ptr<MemoryOperation>, pair_hash >())).first;
  }

  operations->second.insert(std::make_pair(platforms, operation));
}

std::shared_ptr<umpire::op::MemoryOperation>
MemoryOperationRegistry::find(
    const std::string& name,
    std::shared_ptr<strategy::AllocationStrategy>& src_allocator,
    std::shared_ptr<strategy::AllocationStrategy>& dst_allocator)
{
  auto platforms = std::make_pair(
      src_allocator->getPlatform(),
      dst_allocator->getPlatform());

  auto operations = m_operators.find(name);

  if (operations == m_operators.end()) {
    UMPIRE_ERROR("Cannot find operator " << name);
  }

  auto op = operations->second.find(platforms);

  if (op == operations->second.end()) {
    UMPIRE_ERROR("Cannot find operator" << name << " for platforms " << static_cast<int>(platforms.first) << ", " << static_cast<int>(platforms.second));
  }

  return op->second;
}



} // end of namespace op
} // end of namespace umpire
