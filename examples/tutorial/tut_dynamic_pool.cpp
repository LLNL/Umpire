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
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

void allocate_and_deallocate_pool(const std::string& resource)
{
  constexpr size_t SIZE = 1024;

  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator(resource);

  auto pooled_allocator = 
    rm.makeAllocator<umpire::strategy::DynamicPool>(resource + "_pool",
                                                    allocator);

  double* data = static_cast<double*>(
      pooled_allocator.allocate(SIZE*sizeof(double)));

  std::cout << "Allocated " << (SIZE*sizeof(double)) << " bytes using the"
    << pooled_allocator.getName() << " allocator...";

  pooled_allocator.deallocate(data);

  std::cout << " dealocated."
}

int main(int argc, char* argv[]) {
  allocate_and_deallocate_pool("HOST");

#if defined(UMPIRE_ENABLE_CUDA)
  allocate_and_deallocate_pool("DEVICE");
  allocate_and_deallocate_pool("UM");
  allocate_and_deallocate_pool("PINNED");
#endif

  return 0;
}
