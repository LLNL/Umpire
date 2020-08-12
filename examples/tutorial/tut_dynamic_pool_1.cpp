//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPool.hpp"

void allocate_and_deallocate_pool(const std::string& resource)
{
  constexpr std::size_t SIZE = 1024;

  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator(resource);

  auto pooled_allocator = rm.makeAllocator<umpire::strategy::DynamicPool>(
      resource + "_pool", allocator);

  double* data =
      static_cast<double*>(pooled_allocator.allocate(SIZE * sizeof(double)));

  std::cout << "Allocated " << (SIZE * sizeof(double)) << " bytes using the "
            << pooled_allocator.getName() << " allocator...";

  pooled_allocator.deallocate(data);

  std::cout << " deallocated." << std::endl;
}

int main(int, char**)
{
  allocate_and_deallocate_pool("HOST");

#if defined(UMPIRE_ENABLE_CUDA)
  allocate_and_deallocate_pool("DEVICE");
  allocate_and_deallocate_pool("UM");
  allocate_and_deallocate_pool("PINNED");
#endif
#if defined(UMPIRE_ENABLE_HIP)
  allocate_and_deallocate_pool("DEVICE");
  allocate_and_deallocate_pool("PINNED");
#endif

  return 0;
}
