//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPool.hpp"

void allocate_and_deallocate_pool(const std::string& resource,
                                  std::size_t initial_size,
                                  std::size_t min_block_size)
{
  constexpr std::size_t SIZE = 1024;

  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator(resource);

  auto pooled_allocator = rm.makeAllocator<umpire::strategy::DynamicPool>(
      resource + "_pool", allocator, initial_size, /* default = 512Mb*/
      min_block_size /* default = 1Mb */);

  double* data =
      static_cast<double*>(pooled_allocator.allocate(SIZE * sizeof(double)));

  std::cout << "Allocated " << (SIZE * sizeof(double)) << " bytes using the "
            << pooled_allocator.getName() << " allocator...";

  pooled_allocator.deallocate(data);

  std::cout << " deallocated." << std::endl;
}

int main(int, char**)
{
  allocate_and_deallocate_pool("HOST", 65536, 512);
#if defined(UMPIRE_ENABLE_CUDA)
  allocate_and_deallocate_pool("DEVICE", (1024 * 1024 * 1024), (1024 * 1024));
  allocate_and_deallocate_pool("UM", (1024 * 64), 1024);
  allocate_and_deallocate_pool("PINNED", (1024 * 16), 1024);
#endif
#if defined(UMPIRE_ENABLE_HIP)
  allocate_and_deallocate_pool("DEVICE", (1024 * 1024 * 1024), (1024 * 1024));
  allocate_and_deallocate_pool("PINNED", (1024 * 16), 1024);
#endif

  return 0;
}
