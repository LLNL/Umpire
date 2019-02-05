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

void copy_data(double* source_data, size_t size, const std::string& destination)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto dest_allocator = rm.getAllocator(destination);

  double* dest_data = static_cast<double*>(
      dest_allocator.allocate(size*sizeof(double)));

  rm.copy(dest_data, source_data);

  std::cout << "Copied source data (" << source_data << ") to destination "
    << destination << " (" << dest_data << ")" << std::endl;

  dest_allocator.deallocate(dest_data);
}

int main(int, char**) {
  constexpr size_t SIZE = 1024;

  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("HOST");

  double* data = static_cast<double*>(
      allocator.allocate(SIZE*sizeof(double)));

  std::cout << "Allocated " << (SIZE*sizeof(double)) << " bytes using the "
    << allocator.getName() << " allocator." << std::endl;

  std::cout << "Filling with 0.0...";

  for (size_t i = 0; i < SIZE; i++) {
    data[i] = 0.0;
  }

  std::cout << "done." << std::endl;

  copy_data(data, SIZE, "HOST");
#if defined(UMPIRE_ENABLE_CUDA)
  copy_data(data, SIZE, "DEVICE");
  copy_data(data, SIZE, "UM");
  copy_data(data, SIZE, "PINNED");
#endif

  allocator.deallocate(data);

  return 0;
}
