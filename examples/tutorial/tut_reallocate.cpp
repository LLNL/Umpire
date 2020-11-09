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
  constexpr std::size_t REALLOCATED_SIZE = 256;

  auto& rm = umpire::ResourceManager::getInstance();

  const std::string destinations[] = {
    "HOST"
#if defined(UMPIRE_ENABLE_CUDA) || defined(UMPIRE_ENABLE_HIP)
    ,
    "DEVICE",
    "UM",
    "PINNED"
#endif
  };

  for (auto& destination : destinations) {
    auto allocator = rm.getAllocator(destination);
    double* data =
        static_cast<double*>(allocator.allocate(SIZE * sizeof(double)));

    std::cout << "Allocated " << (SIZE * sizeof(double)) << " bytes using the "
              << allocator.getName() << " allocator." << std::endl;

    std::cout << "Reallocating data (" << data << ") to size "
              << REALLOCATED_SIZE << "...";

    // _sphinx_tag_tut_realloc_start
    data = static_cast<double*>(rm.reallocate(data, REALLOCATED_SIZE));
    // _sphinx_tag_tut_realloc_end

    std::cout << "done.  Reallocated data (" << data << ")" << std::endl;

    allocator.deallocate(data);
  }

  return 0;
}
