//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/config.hpp"

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

//
// For debugging purposes, this program uses the number of command line
// as a flag to indicate the mode it should run in.  The modes are:
//
// 1) If run with no argument (ac==1), it will run as an MPI program.
// 2) If run with 1 argument (ac==2), it will run as the parent non-mpi program.
// 3) If run with 2 arguments (ac==3), it will run as the child non-mpi program.
//
// This will allow someone to launch this program as a Parent and Child in two
// separate debugger session windows (possible in gdb or vscode).  When running
// in the debugger session windows, no MPI will be used and the debugger must
// be used for synchronization (by setting breakpoints).
//
int main(
  int
  ac
  ,
  char**
  av
)
{
  const bool use_mpi{ ac == 1 };
  const bool i_am_parent{ ac == 2 };

  if (use_mpi) {
    MPI_Init(&ac, &av);
  }

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
  MPI_Comm shared_allocator_comm;
  int foreman_rank{0};
  int shared_rank{0};

  if (use_mpi) {
    shared_allocator_comm = umpire::get_communicator_for_allocator(
                                      node_allocator, MPI_COMM_WORLD);
    MPI_Comm_rank(shared_allocator_comm, &shared_rank);
  }
  else
  { // Running non-mpi in debugger
    shared_rank = i_am_parent ? foreman_rank : foreman_rank + 1;
  }

  //
  // Allocate shared memory
  //
  void* ptr{ node_allocator.allocate("allocation_name_2", sizeof(uint64_t)) };
  uint64_t* data{ static_cast<uint64_t*>(ptr) };

  if ( shared_rank == foreman_rank )
    *data = 0xDEADBEEF;

  if (use_mpi) {
    MPI_Barrier(shared_allocator_comm);
  }
  else {
    if ( !i_am_parent ) {
      shared_rank++;    // Set a breakpoint here to synchronize
    }
  }

  UMPIRE_ASSERT(*data == 0xDEADBEEF);

  node_allocator.deallocate(ptr);

  if (use_mpi) {
    MPI_Finalize();
  }

  return 0;
}

