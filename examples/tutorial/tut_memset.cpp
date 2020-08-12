//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

int main(int, char**)
{
  constexpr std::size_t SIZE = 1024;

  auto& rm = umpire::ResourceManager::getInstance();

  const std::string destinations[] = {
    "HOST"
#if defined(UMPIRE_ENABLE_CUDA)
    ,
    "DEVICE",
    "UM",
    "PINNED"
#endif
#if defined(UMPIRE_ENABLE_HIP)
    ,
    "DEVICE",
    "PINNED"
#endif
  };

  for (auto& destination : destinations) {
    auto allocator = rm.getAllocator(destination);
    double* data =
        static_cast<double*>(allocator.allocate(SIZE * sizeof(double)));

    std::cout << "Allocated " << (SIZE * sizeof(double)) << " bytes using the "
              << allocator.getName() << " allocator." << std::endl;

    rm.memset(data, 0);

    std::cout << "Set data from " << destination << " (" << data << ") to 0."
              << std::endl;

    allocator.deallocate(data);
  }

  return 0;
}
