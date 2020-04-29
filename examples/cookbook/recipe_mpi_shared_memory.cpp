//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

#include <iostream>

#include "mpi.h"

void initData(void* shared_ptr)
{
  uint64_t* rval = reinterpret_cast<uint64_t*>(shared_ptr);

  *rval = 12345678;
}

int main(int ac, char** av) {
  MPI_Init(&ac, &av);

  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("MPI_SHARED_MEM");
  auto ptr = allocator.allocate(64);

  //
  // Is collective call, only the foreman will call initData
  // 
  // memset implementation for MPI resource will have all ranks
  // synchronize on a fence until foreman is done with function call.
  //
  rm.memset(ptr, 64, initData);

  uint64_t* Data = reinterpret_cast<uint64_t*>(ptr);

  std::cout << "Data pointer: " << Data << " is contains: " << *Data << std::endl;

  allocator.deallocate(ptr);

  MPI_Finalize();

  return 0;
}
