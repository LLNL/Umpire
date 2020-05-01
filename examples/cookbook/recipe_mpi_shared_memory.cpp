//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/NamedAllocator.hpp"

#include <iostream>
#include <string>

#include "mpi.h"

void initData(void* shared_ptr)
{
  uint64_t* rval = reinterpret_cast<uint64_t*>(shared_ptr);

  *rval = 12345678;
}

int main(int ac, char** av) {
  MPI_Init(&ac, &av);

  auto& rm = umpire::ResourceManager::getInstance();
  auto shared_allocator = rm.getAllocator("MPI_SHARED_MEM");


  //
  // 1. Determine whether allocator is for shared memory
  //
  UMPIRE_ASSERT(shared_allocator.getPlatform() == umpire::Platform::mpi_shmem);

  //
  // 2. Allow for unnamed allocation/deallocation
  //
  {
    auto ptr = shared_allocator.allocate(64);
    shared_allocator.deallocate(ptr);
  }

  //
  // 3. Allow for named allocation/deallocation
  //
  {
    auto named_allocator = umpire::NamedAllocator{shared_allocator};
    auto ptr = named_allocator.allocate(std::string{"Named Allocation"}, 128);
    std::cout << "ptr=" << ptr << std::endl;
    named_allocator.deallocate(ptr);
  }

  //
  // 4. Allow for finding allocation by name
  //
  {
    auto named_allocator{umpire::NamedAllocator{shared_allocator}};
    auto ptr{named_allocator.allocate(std::string{"Named Allocation"}, 128)};
    auto ptr2{named_allocator.get_allocation_by_name(std::string{"Named Allocation"})};
    std::cout << "ptr=" << ptr << ", ptr2=" << ptr2 << std::endl;
    UMPIRE_ASSERT(ptr != nullptr && ptr == ptr2);
    named_allocator.deallocate(ptr);
  }

  //
  // 5. Determine whether I own memory (is my rank the designated owner?)
  //

  //
  // 6. Synchronize on the allocation
  //

  //
  // 7. Set up different rank as owner (foreman)
  //

  //
  // 8. Set the communicator group for this node (construction?)
  //
  MPI_Finalize();

  return 0;
}
