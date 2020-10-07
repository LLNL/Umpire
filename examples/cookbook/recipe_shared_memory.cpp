//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/resource/BoostMemoryResource.hpp"
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

  //
  // Create/Attach to our two named, sized allocators
  //
  auto& rm = umpire::ResourceManager::getInstance();
  auto traits{umpire::get_default_resource_traits("SHARED")};
  traits.size = 1024*1024;
  traits.scope = umpire::MemoryResourceTraits::shared_scope::NODE;
  auto allocator_1MiB{rm.makeResource("SHARED::1MiB_allocator", traits)};
  UMPIRE_ASSERT(allocator_1MiB.getAllocationStrategy()->getTraits().resource
                        == umpire::MemoryResourceTraits::resource_type::SHARED);

  traits.size *= 1024;
  auto allocator_1GiB{rm.makeResource("SHARED::1GiB_allocator", traits)};
  UMPIRE_ASSERT(allocator_1GiB.getAllocationStrategy()->getTraits().resource
                        == umpire::MemoryResourceTraits::resource_type::SHARED);

  auto shared_allocator_comm = umpire::get_communicator_for_allocator(
      allocator_1MiB, MPI_COMM_WORLD);

  int shared_rank;
  MPI_Comm_rank(shared_allocator_comm, &shared_rank);

  //
  // Allocate and initialize if foreman, wait/attach if not
  //
  {
    void* ptr{nullptr};
    if (shared_rank == foreman_rank) {
      ptr = allocator_1MiB.allocate("allocation_name_1", sizeof(uint64_t));
      uint64_t* data{static_cast<uint64_t*>(ptr)};
      *data = 0xDEADBEEF;
    }

    MPI_Barrier(shared_allocator_comm);

    if ( shared_rank != foreman_rank )
      ptr = allocator_1MiB.allocate("allocation_name_1", sizeof(uint64_t));

    uint64_t* data{static_cast<uint64_t*>(ptr)};
    UMPIRE_ASSERT(*data == 0xDEADBEEF);

    MPI_Barrier(shared_allocator_comm);
    allocator_1MiB.deallocate(ptr);
  }

  //
  // Race for allocation/attach, foreman write, others wait
  //
  {
    void* ptr{allocator_1GiB.allocate("allocation_name_2", sizeof(uint64_t))};
    uint64_t* data{static_cast<uint64_t*>(ptr)};

    if ( shared_rank == foreman_rank )
      *data = 0xDEADBEEF;

    UMPIRE_ASSERT(
      umpire::find_pointer_from_name(allocator_1GiB, "allocation_name_2") == ptr);

    MPI_Barrier(shared_allocator_comm);

    UMPIRE_ASSERT(*data == 0xDEADBEEF);

    MPI_Barrier(shared_allocator_comm);

    allocator_1GiB.deallocate(ptr);
  }

  {
    auto traits{umpire::get_default_resource_traits("SHARED")};
    traits.size = 1024*1024;
    traits.scope = umpire::MemoryResourceTraits::shared_scope::SOCKET;
    try {
      auto socket_allocator{rm.makeResource("SHARED::socket", traits)};
      void* ptr{
        socket_allocator.allocate("socket_allocation_1", sizeof(uint64_t))};
      socket_allocator.deallocate(ptr);
    } catch (...) {
      std::cout << "Cannot create resource" << std::endl;
    }
  }

  MPI_Finalize();

  return 0;
}
