//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/ThreadSafeAllocator.hpp"

int main(int, char**)
{
  // _umpire_tut_thread_safe_start
  auto& rm = umpire::ResourceManager::getInstance();

  auto pool = rm.makeAllocator<umpire::strategy::DynamicPool>(
      "pool", rm.getAllocator("HOST"));

  auto thread_safe_pool =
      rm.makeAllocator<umpire::strategy::ThreadSafeAllocator>(
          "thread_safe_pool", pool);
  // _umpire_tut_thread_safe_end

  auto allocation = thread_safe_pool.allocate(256);
  thread_safe_pool.deallocate(allocation);

  return 0;
}
