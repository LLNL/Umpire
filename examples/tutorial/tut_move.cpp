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

double* move_data(double* source_data, const std::string& destination)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto dest_allocator = rm.getAllocator(destination);

  double* dest_data = static_cast<double*>(
      dest_allocator.allocate(size*sizeof(double)));

  std::cout << "Moved source data (" << source_data << ") to destination ";

  double* dest_data = static_cast<double*>(
      rm.move(source_data, dest_allocator));

  std::cout << destination << " (" << dest_data << ")" << std::endl;

  return dest_data;
}

int main(int argc, char* argv[]) {
  constexpr size_t SIZE = 1024;

  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("HOST");

  double* data = static_cast<double*>(
      allocator.allocate(SIZE*sizeof(double)));

  std::cout << "Allocated " << (SIZE*sizeof(double)) << " bytes using the"
    << allocator.getName() << " allocator." << std::endl;

  std::cout << "Filling with 0.0...";

  for (size_t i = 0; i < SIZE; i++) {
    data[i] = 0.0;
  }

  std::cout << "done." << std::endl;

  data = move_data(data, "HOST");
#if defined(UMPIRE_ENABLE_CUDA)
  data = move_data(data, "DEVICE");
  data = move_data(data, "UM");
  data = move_data(data, "PINNED");
#endif

  rm.deallocate(data);

  return 0;
}
