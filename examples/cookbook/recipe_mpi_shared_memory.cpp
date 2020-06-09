//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/SharedMemoryAllocator.hpp"

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include "mpi.h"

void print(bool i_am_foreman, std::string s)
{
  if ( i_am_foreman ) {
    std::cout << s << std::endl;
  }
}

int main(int ac, char** av) {
  MPI_Init(&ac, &av);

  auto& rm = umpire::ResourceManager::getInstance();
  umpire::SharedMemoryAllocator allocator = rm.getSharedMemoryAllocator("MPI_SHARED_MEM");

  print(allocator.is_foreman(), "1.) Determine whether I am the foreman");

  print(allocator.is_foreman(), "2.) Confirm our allocator is indeed mpi_shmem resource");
  UMPIRE_ASSERT(allocator.getPlatform() == umpire::Platform::mpi_shmem);

  {
    print(allocator.is_foreman(), "3.) Unnamed allocation/deallocation");
    auto ptr{allocator.allocate(64)};
    UMPIRE_ASSERT(ptr != nullptr);
    allocator.deallocate(ptr);
  }

  {
    std::string name{"Named Allocation"};

    print(allocator.is_foreman(), "4.) Named allocation/deallocation");
    auto ptr{allocator.allocate(name, 128)};
    UMPIRE_ASSERT(ptr != nullptr);

    print(allocator.is_foreman(), "5.) Allow for finding allocation by name");
    auto ptr2{allocator.get_allocation_by_name(name)};
    UMPIRE_ASSERT(ptr == ptr2);

    allocator.deallocate(ptr);
  }

  {
    auto ptr = allocator.allocate(sizeof(uint64_t));
    uint64_t* data{static_cast<uint64_t*>(ptr)};

    print(allocator.is_foreman(), "6.) Modify memory as foreman (3 second delay)");
    if ( allocator.is_foreman() ) {
      std::this_thread::sleep_for (std::chrono::seconds(3));
      *data = 0xDEADBEEF;
    }

    print(allocator.is_foreman(), "7.) Synchronize");
    allocator.synchronize();
    UMPIRE_ASSERT(*data == 0xDEADBEEF);

    allocator.deallocate(ptr);
  }

/**
  //
  // 7. Set up different rank as owner (foreman)
  //

  //
  // 8. Set the communicator group for this node (construction?)
  //
**/

  MPI_Finalize();

  return 0;
}
