//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <fstream>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/strategy/DynamicPool.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("HOST");
  auto pooled_allocator = rm.makeAllocator<umpire::strategy::DynamicPool>(
      "HOST_POOL", allocator, 1024 * 16);

  void* a[4];
  for (int i = 0; i < 4; ++i)
    a[i] = pooled_allocator.allocate(1024);

  // Create fragmentation
  pooled_allocator.deallocate(a[2]);
  a[2] = pooled_allocator.allocate(1024 * 2);

  // Output the records from the underlying host allocator
  {
    std::ofstream out("allocator.log");
    umpire::print_allocator_records(allocator, out);
    out.close();
  }

  // Output the records from the pooled allocator
  {
    std::ofstream out("pooled_allocator.log");
    umpire::print_allocator_records(pooled_allocator, out);
    out.close();
  }

  for (int i = 0; i < 4; ++i)
    pooled_allocator.deallocate(a[i]);

  // Visualize this using the python script. Example usage:
  // tools/analysis/plot_allocations allocator.log gray 0.2 pooled_allocator.log
  // purple 0.8

  return 0;
}
