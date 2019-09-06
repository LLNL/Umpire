//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

int main(int, char**) {
  constexpr std::size_t SIZE = 1024;

  auto& rm = umpire::ResourceManager::getInstance();

  const std::string destinations[] = {
    "HOST"
#if defined(UMPIRE_ENABLE_CUDA)
      , "DEVCIE"
      , "UM"
      , "PINNED"
#endif
#if defined(UMPIRE_ENABLE_HIP)
      , "DEVCIE"
      , "PINNED"
#endif
  };

  for (auto& destination : destinations) {
    auto allocator = rm.getAllocator(destination);
    double* data = static_cast<double*>(
        allocator.allocate(SIZE*sizeof(double)));

    std::cout << "Allocated " << (SIZE*sizeof(double)) << " bytes using the "
      << allocator.getName() << " allocator." << std::endl;

    auto found_allocator = rm.getAllocator(data);

    std::cout << "According to the ResourceManager, the Allocator used is "
      << found_allocator.getName()
      << ", which has the Platform "
      << static_cast<int>(found_allocator.getPlatform()) << std::endl;

    std::cout << "The size of the allocation is << "
      << found_allocator.getSize(data) << std::endl;

    allocator.deallocate(data);
  }

  return 0;
}
