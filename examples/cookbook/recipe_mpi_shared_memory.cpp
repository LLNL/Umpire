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

uint64_t* initData(void* shared_ptr)
{
  uint64_t* rval = reinterpret_cast<uint64_t*>(shared_ptr);

  *rval = 0x12345678;

  return rval;
}

int main(int ac, char** av) {
  MPI_Init(&ac, &av);

  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("MPI_SHARED_MEM");

  auto ptr = allocator.allocate(64);

  uint64_t* Data = initData(ptr);

  std::cout << "Data pointer: " << Data << " is contains: " << *Data << std::endl;

  allocator.deallocate(ptr);

  MPI_Finalize();

  return 0;
}
