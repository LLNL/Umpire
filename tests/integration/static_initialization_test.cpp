//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPoolList.hpp"
#include "umpire/strategy/FixedPool.hpp"

#if defined(UMPIRE_ENABLE_HEADER_INTROSPECTION)
#include "umpire/util/allocation_metadata.hpp"
#endif

static auto& rm = umpire::ResourceManager::getInstance();

static auto alloc = rm.getAllocator("HOST");
static auto dyn_pool_alloc = rm.makeAllocator<umpire::strategy::DynamicPoolList>("host_dyn_pool", alloc);
#if defined(UMPIRE_ENABLE_HEADER_INTROSPECTION)
// In header introspection mode, FixedPool must accommodate both user data and the aligned header
static auto fixed_pool_alloc = rm.makeAllocator<umpire::strategy::FixedPool>(
    "host_fixed_pool", alloc, 512 + sizeof(umpire::util::allocation_header<umpire::util::AllocationRecord>));
#else
static auto fixed_pool_alloc = rm.makeAllocator<umpire::strategy::FixedPool>("host_fixed_pool", alloc, 512);
#endif

static void* alloc_data{alloc.allocate(512)};
static void* dyn_pool_data{dyn_pool_alloc.allocate(512)};
static void* fixed_pool_data{fixed_pool_alloc.allocate(512)};

int main()
{
  alloc.deallocate(alloc_data);
  dyn_pool_alloc.deallocate(dyn_pool_data);
  fixed_pool_alloc.deallocate(fixed_pool_data);
  return 0;
}
