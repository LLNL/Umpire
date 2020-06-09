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

void initData(void* shared_ptr)
{
  uint64_t* rval = reinterpret_cast<uint64_t*>(shared_ptr);

  *rval = 12345678;
}

int main(int ac, char** av) {
  MPI_Init(&ac, &av);

  auto& rm = umpire::ResourceManager::getInstance();
  umpire::SharedMemoryAllocator shared_allocator = rm.getSharedMemoryAllocator("MPI_SHARED_MEM");

  //
  // 1. Determine whether allocator is for shared memory
  //
  UMPIRE_ASSERT(shared_allocator.getPlatform() == umpire::Platform::mpi_shmem);

  //
  // 2. Allow for unnamed allocation/deallocation
  //
  {
    auto ptr = shared_allocator.allocate(64);
    UMPIRE_ASSERT(ptr != nullptr);
    shared_allocator.deallocate(ptr);
  }

  //
  // 3. Allow for named allocation/deallocation
  //
  {
    auto ptr = shared_allocator.allocate(std::string{"Named Allocation"}, 128);
    UMPIRE_ASSERT(ptr != nullptr);
    shared_allocator.deallocate(ptr);
  }

  //
  // 4. Allow for finding allocation by name
  //
  {
    std::string name{"Named Allocation"};
    auto ptr{shared_allocator.allocate(name, 128)};
    auto ptr2 = shared_allocator.get_allocation_by_name(name);
    std::cout << "ptr=" << ptr << ", ptr2=" << ptr2 << std::endl;
    UMPIRE_ASSERT(ptr != nullptr && ptr == ptr2);
    shared_allocator.deallocate(ptr);
  }

  {
    auto ptr = shared_allocator.allocate(sizeof(uint64_t));
    uint64_t* data{static_cast<uint64_t*>(ptr)};

    //
    // 5. Determine whether I own memory (is my rank the designated owner?)
    //
    if ( shared_allocator.is_foreman() ) {
      // Delay a second to make sure everyone has a chance to test synchronization
      std::this_thread::sleep_for (std::chrono::seconds(5));

      *data = 0xDEADBEEF;
    }

    //
    // 6. Synchronize on the allocation
    //
    shared_allocator.synchronize();
    UMPIRE_ASSERT(*data == 0xDEADBEEF);

    shared_allocator.deallocate(ptr);
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
