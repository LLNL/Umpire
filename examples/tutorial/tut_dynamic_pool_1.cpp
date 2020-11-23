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
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator(resource);

  // _sphinx_tag_tut_makepool_start
  auto pooled_allocator = rm.makeAllocator<umpire::strategy::DynamicPool>(
      resource + "_pool", allocator);
  // _sphinx_tag_tut_makepool_end

  constexpr std::size_t SIZE = 1024;

  // _sphinx_tag_tut_allocate_start
  double* data =
      static_cast<double*>(pooled_allocator.allocate(SIZE * sizeof(double)));
  // _sphinx_tag_tut_allocate_end

  std::cout << "Allocated " << (SIZE * sizeof(double)) << " bytes using the "
            << pooled_allocator.getName() << " allocator...";

  // _sphinx_tag_tut_deallocate_start
  pooled_allocator.deallocate(data);
  // _sphinx_tag_tut_deallocate_end

  std::cout << " deallocated." << std::endl;
}

int main(int, char**)
{
  // _sphinx_tag_tut_anyallocator_start
  allocate_and_deallocate_pool("HOST");

#if defined(UMPIRE_ENABLE_DEVICE)
  allocate_and_deallocate_pool("DEVICE");
#endif
#if defined(UMPIRE_ENABLE_UM)
  allocate_and_deallocate_pool("UM");
#endif
#if defined(UMPIRE_ENABLE_PINNED)
  allocate_and_deallocate_pool("PINNED");
#endif
  // _sphinx_tag_tut_anyallocator_end

  return 0;
}
