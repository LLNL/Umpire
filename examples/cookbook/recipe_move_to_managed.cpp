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

  //
  // Allocate host data
  //
  double* host_data =
      static_cast<double*>(allocator.allocate(SIZE * sizeof(double)));

  //
  // Move data to unified memory
  //
  auto um_allocator = rm.getAllocator("UM");
  // _sphinx_tag_tut_move_host_to_managed_start
  double* um_data = static_cast<double*>(rm.move(host_data, um_allocator));
  // _sphinx_tag_tut_move_host_to_managed_end

  //
  // Deallocate um_data, host_data is already deallocated by move operation.
  //
  rm.deallocate(um_data);

  return 0;
}
