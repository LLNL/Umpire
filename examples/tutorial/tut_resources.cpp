//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

void allocate_and_deallocate(const std::string& resource)
{
  constexpr std::size_t SIZE = 1024;

  auto& rm = umpire::ResourceManager::getInstance();

  umpire::Allocator allocator = rm.getAllocator(resource);

  double* data =
      static_cast<double*>(allocator.allocate(SIZE * sizeof(double)));

  std::cout << "Allocated " << (SIZE * sizeof(double)) << " bytes using the "
            << allocator.getName() << " allocator...";

  allocator.deallocate(data);

  std::cout << " deallocated." << std::endl;
}

int main(int, char**)
{
  allocate_and_deallocate("HOST");

#if defined(UMPIRE_ENABLE_CUDA)
  allocate_and_deallocate("DEVICE");
  allocate_and_deallocate("UM");
  allocate_and_deallocate("PINNED");
#endif
#if defined(UMPIRE_ENABLE_HIP)
  allocate_and_deallocate("DEVICE");
  allocate_and_deallocate("PINNED");
#endif

  return 0;
}
