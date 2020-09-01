//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPool.hpp"

static auto& rm = umpire::ResourceManager::getInstance();

static auto alloc = rm.getAllocator("HOST");
static auto dyn_pool_alloc =
    rm.makeAllocator<umpire::strategy::DynamicPool>("host_dyn_pool", alloc);
static auto fixed_pool_alloc = rm.makeAllocator<umpire::strategy::FixedPool>(
    "host_fixed_pool", alloc, 512);

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
