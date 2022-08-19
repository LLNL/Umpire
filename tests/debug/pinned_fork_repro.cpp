//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <stdlib.h>
#include "mpi.h"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/QuickPool.hpp"

static auto& rm = umpire::ResourceManager::getInstance();
static auto pool_alloc = rm.makeAllocator<umpire::strategy::QuickPool>("quickpool", rm.getAllocator("PINNED"));
static void* dyn_pool_data{pool_alloc.allocate(512)};

int main(int ac, char** av)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto alloc = rm.getAllocator("quickpool");
  char* my_array = static_cast<char*>(alloc.allocate(1024));

  MPI_Init(&ac, &av);
  system("hostname");
  MPI_Finalize();

  alloc.deallocate(my_array);
}
