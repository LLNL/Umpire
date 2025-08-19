#include <mpi.h>

#include <iostream>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/config.hpp"
#include "umpire/resource/HostMpi3SharedMemoryResource.hpp"
#include "umpire/strategy/NamedAllocationStrategy.hpp"
#include "umpire/util/MemoryResourceTraits.hpp"

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  auto& rm = umpire::ResourceManager::getInstance();

  // Use MPI3 shared memory resource
  // Note: Could also use "SHARED"
  auto traits = umpire::get_default_resource_traits("SHARED::MPI3");
  traits.size = 1 * 1024 * 1024; // 1 MB

  // Node scope is required for mpi3 shared memory
  traits.scope = umpire::MemoryResourceTraits::shared_scope::node;

  // Create allocator using MPI3 shared memory
  auto mpi3_shm_allocator = rm.makeResource("SHARED::mpi3_alloc", traits);

  // Get communicator for the allocator
  MPI_Comm shm_comm = umpire::get_communicator_for_allocator(mpi3_shm_allocator, MPI_COMM_WORLD);

  int rank = 0;
  MPI_Comm_rank(shm_comm, &rank);

  // Allocate shared memory, doesn't need a name for allocation
  uint64_t* data = static_cast<uint64_t*>(mpi3_shm_allocator.allocate(sizeof(uint64_t)));

  if (rank == 0) {
    *data = 0xCAFEBABE;
  }

  MPI_Barrier(shm_comm);

  // All ranks should see the same value
  std::cout << "Rank " << rank << " sees value: " << std::hex << *data << std::endl;

  mpi3_shm_allocator.deallocate(data);

  umpire::cleanup_cached_communicators();
  MPI_Finalize();

  return 0;
}
