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
#include "umpire/op/HostReallocateOperation.hpp"

#include <cstdlib>

#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void HostReallocateOperation::transform(
    void* src_ptr,
    void** dst_ptr,
    util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    util::AllocationRecord *dst_allocation,
    size_t length)
{
  auto allocator = dst_allocation->m_strategy;

  ResourceManager::getInstance().deregisterAllocation(src_ptr);

  *dst_ptr = ::realloc(src_ptr, length);

  ResourceManager::getInstance().registerAllocation(
      *dst_ptr,
      new util::AllocationRecord{*dst_ptr, length, allocator});
}

} // end of namespace op
} // end of namespace umpire
