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
  auto allocator = rm.getAllocator("HOST");

  /*
   * Allocate host data
   */
  double* host_data =
      static_cast<double*>(allocator.allocate(SIZE * sizeof(double)));

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
