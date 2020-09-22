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

namespace {
  //
  // Umpire provides Shared Memory, Umpire user provides sync mechanism.
  //
  struct SynchronizationMechanism {
    public:
      SynchronizationMechanism(MPI_Comm _comm, int _foreman) : comm{_comm}, foreman{_foreman} {
        MPI_Comm_rank(comm, &my_rank);
      }

      bool is_foreman() { return my_rank == foreman; }
      // void synchronize() { MPI_Barrier(comm); }
      void synchronize() { MPI_Barrier(comm); }

      int my_rank;
    private:
      MPI_Comm comm;
      int foreman;
  };
} // end of anonymous namespace

int main(int ac, char** av)
{
  MPI_Init(&ac, &av);

  //
  // Initialize our synchronization object
  //
  const int foreman_rank{0};
  MPI_Comm shmcomm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
  int shmrank;
  MPI_Comm_rank(shmcomm, &shmrank);
  SynchronizationMechanism sync{shmcomm, foreman_rank};

  //
  // Create/Attach to our two named, sized allocators
  //
  auto& rm = umpire::ResourceManager::getInstance();
  auto traits{umpire::get_default_resource_traits("SHARED")};
  traits.size = 1024*1024;
  auto allocator_1KiB{rm.makeResource("SHARED::1KiB_allocator", traits)};
  UMPIRE_ASSERT(allocator_1KiB.getAllocationStrategy()->getTraits().resource
                        == umpire::MemoryResourceTraits::resource_type::SHARED);

  traits.size *= 1024;
  auto allocator_1MiB{rm.makeResource("SHARED::1MiB_allocator", traits)};
  UMPIRE_ASSERT(allocator_1MiB.getAllocationStrategy()->getTraits().resource
                        == umpire::MemoryResourceTraits::resource_type::SHARED);

  //
  // Allocate and initialize if foreman, wait/attach if not
  //
  {
    void* ptr{nullptr};
    if ( sync.is_foreman() ) {
      ptr = allocator_1KiB.allocate("allocation_name_1", 100 * sizeof(uint64_t));
      uint64_t* data{static_cast<uint64_t*>(ptr)};
      *data = 0xDEADBEEF;
    }

    sync.synchronize();

    if ( ! sync.is_foreman() )
      ptr = allocator_1KiB.allocate("allocation_name_1", 100 * sizeof(uint64_t));

    uint64_t* data{static_cast<uint64_t*>(ptr)};
    UMPIRE_ASSERT(*data == 0xDEADBEEF);

    sync.synchronize();
    allocator_1KiB.deallocate(ptr);
  }

  //
  // Race for allocation/attach, foreman write, others wait
  //
  {
    void* ptr{allocator_1MiB.allocate("allocation_name_2", sizeof(uint64_t))};
    uint64_t* data{static_cast<uint64_t*>(ptr)};

    if ( sync.is_foreman() )
      *data = 0xDEADBEEF;

    UMPIRE_ASSERT(
      umpire::find_pointer_from_name(allocator_1MiB, "allocation_name_2") == ptr );

    sync.synchronize();

    UMPIRE_ASSERT(*data == 0xDEADBEEF);

    sync.synchronize();

    allocator_1MiB.deallocate(ptr);
  }

  MPI_Finalize();

  return 0;
}
