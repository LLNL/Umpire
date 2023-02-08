//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPoolList.hpp"

void allocate_and_deallocate_pool(const std::string& resource, std::size_t initial_size, std::size_t min_block_size)
{
  constexpr std::size_t SIZE = 1024;

  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator(resource);

  // _sphinx_tag_tut_allocator_tuning_start
  auto pooled_allocator = rm.makeAllocator<umpire::strategy::DynamicPoolList>(resource + "_pool", allocator,
                                                                              initial_size, /* default = 512Mb*/
                                                                              min_block_size /* default = 1Mb */);
  // _sphinx_tag_tut_allocator_tuning_end

  double* data = static_cast<double*>(pooled_allocator.allocate(SIZE * sizeof(double)));

  std::cout << "Allocated " << (SIZE * sizeof(double)) << " bytes using the " << pooled_allocator.getName()
            << " allocator...";

  pooled_allocator.deallocate(data);

  std::cout << " deallocated." << std::endl;
}

int main(int, char**)
{
  // _sphinx_tag_tut_device_sized_pool_start
  allocate_and_deallocate_pool("HOST", 65536, 512);
#if defined(UMPIRE_ENABLE_DEVICE)
  allocate_and_deallocate_pool("DEVICE", (1024 * 1024 * 1024), (1024 * 1024));
#endif
#if defined(UMPIRE_ENABLE_UM)
  allocate_and_deallocate_pool("UM", (1024 * 64), 1024);
#endif
#if defined(UMPIRE_ENABLE_PINNED)
  allocate_and_deallocate_pool("PINNED", (1024 * 16), 1024);
#endif
  // _sphinx_tag_tut_device_sized_pool_end

  return 0;
}
