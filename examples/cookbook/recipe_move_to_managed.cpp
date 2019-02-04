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

  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("HOST");

  /*
   * Allocate host data
   */
  double* host_data = static_cast<double*>(
      allocator.allocate(SIZE*sizeof(double)));

  /*
   * Move data to unified memory
   */
  auto um_allocator = rm.getAllocator("UM");
  double* um_data = static_cast<double*>(rm.move(host_data, um_allocator));

  /*
   * Deallocate um_data, host_data is already deallocated by move operation.
   */
  rm.deallocate(um_data);

  return 0;
}
