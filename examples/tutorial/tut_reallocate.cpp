//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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

int main(int, char**) {
  constexpr size_t SIZE = 1024;
  constexpr size_t REALLOCATED_SIZE = 256;

  auto& rm = umpire::ResourceManager::getInstance();

  const std::string destinations[] = {
    "HOST"
#if defined(UMPIRE_ENABLE_CUDA)
      , "DEVICE"
      , "UM"
      , "PINNED"
#endif
  };

  for (auto& destination : destinations) {
    auto allocator = rm.getAllocator(destination);
    double* data = static_cast<double*>(
        allocator.allocate(SIZE*sizeof(double)));

    std::cout << "Allocated " << (SIZE*sizeof(double)) << " bytes using the "
      << allocator.getName() << " allocator." << std::endl;

    std::cout << "Reallocating data (" << data << ") to size " 
      << REALLOCATED_SIZE << "...";

    data = static_cast<double*>(rm.reallocate(data, REALLOCATED_SIZE));

    std::cout << "done.  Reallocated data (" << data << ")" << std::endl;

    allocator.deallocate(data);
  }

  return 0;
}
