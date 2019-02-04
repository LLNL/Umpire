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
#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"

#include <iostream>

int main() {
  auto &rm = umpire::ResourceManager::getInstance();
  auto alloc = rm.getAllocator("HOST");

  const int size = 100;

  double* my_array = static_cast<double*>(alloc.allocate(100 * sizeof(double)));

  for (int i = 0; i < size; i++) {
    my_array[i] = static_cast<double>(i);
  }

  for (int i = 0; i < size; i++) {
    std::cout << my_array[i] << " should be " << i << std::endl;
  }

  alloc.deallocate(my_array);
}
