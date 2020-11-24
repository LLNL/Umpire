//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/resource/HostSharedMemoryResource.hpp"
#include "umpire/util/MemoryResourceTraits.hpp"

#include "mpi.h"

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

int main(int ac, char** av)
{
  MPI_Init(&ac, &av);
  const int foreman_rank{0};

  auto& rm = umpire::ResourceManager::getInstance();

  //
  // Set up the traits for the allocator
  //
  auto traits{umpire::get_default_resource_traits("SHARED")};
  traits.size = 1*1024*1024;  // Maximum size of this Allocator

  //
  // Default scope for allocator is NODE.  SOCKET is another option of interest
  //
  traits.scope = umpire::MemoryResourceTraits::shared_scope::node;  // default

  //
  // Create (or attach to) the allocator
  //
  auto node_allocator{rm.makeResource("SHARED::node_allocator", traits)};

  //
  // Resource of this allocator is SHARED
  //
  UMPIRE_ASSERT(node_allocator.getAllocationStrategy()->getTraits().resource
                        == umpire::MemoryResourceTraits::resource_type::shared);

  //
  // Get communicator for this allocator
  //
  auto shared_allocator_comm = umpire::get_communicator_for_allocator(
      node_allocator, MPI_COMM_WORLD);

  int shared_rank;

  MPI_Comm_rank(shared_allocator_comm, &shared_rank);

  //
  // Allocate shared memory
  //
  void* ptr{node_allocator.allocate("allocation_name_2", sizeof(uint64_t))};
  uint64_t* data{static_cast<uint64_t*>(ptr)};

  if ( shared_rank == foreman_rank )
    *data = 0xDEADBEEF;

  //
  // Allocation pointers may also be obtained given the allocation name
  //
  UMPIRE_ASSERT(
    umpire::find_pointer_from_name(node_allocator, "allocation_name_2") == ptr);

  MPI_Barrier(shared_allocator_comm);

  UMPIRE_ASSERT(*data == 0xDEADBEEF);

  node_allocator.deallocate(ptr);

  MPI_Finalize();

  return 0;
}
