//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/resource/BoostMemoryResource.hpp"
#include "umpire/util/MemoryResourceTraits.hpp"

#include "mpi.h"

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

bool is_foreman()
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank == 0;
}

void synchronize()
{
  MPI_Barrier(MPI_COMM_WORLD);
}

void print(bool i_am_foreman, std::string s)
{
  if ( i_am_foreman ) {
    std::cout << s << std::endl;
  }
}

int main(int ac, char** av)
{
  MPI_Init(&ac, &av);
  std::string name{"shared_allocation_1"};

  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator{rm.getAllocator("SHARED")};
  auto allocator_two{rm.makeResource("SHARED::extra")};

  print(is_foreman(), "1.) Confirm our allocator is indeed a shared memory resource");
  UMPIRE_ASSERT( "Trait mismatch" &&
    allocator.getAllocationStrategy()->getTraits().resource == umpire::MemoryResourceTraits::resource_type::SHARED);

  print(is_foreman(), "2.) Named allocation/deallocation");

  auto ptr = allocator.allocate(name, 10 * sizeof(uint64_t));
  auto ptr = allocator_two.allocate(name, 10 * sizeof(uint64_t));
  uint64_t* data{static_cast<uint64_t*>(ptr)};

  print(is_foreman(), "3.) Modify memory as foreman (3 second delay)");

  if ( is_foreman() ) {
    std::this_thread::sleep_for (std::chrono::seconds(3));
    *data = 0xDEADBEEF;
  }

  print(is_foreman(), "4.) Synchronize");
  synchronize();

  UMPIRE_ASSERT(*data == 0xDEADBEEF);

  allocator.deallocate(ptr);

  MPI_Finalize();

  return 0;
}