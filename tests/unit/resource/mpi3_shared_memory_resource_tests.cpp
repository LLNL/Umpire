//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "gtest/gtest.h"
#include "mpi.h"
#include "umpire/Umpire.hpp"

class MPISharedMemoryTest : public ::testing::Test {
 protected:
  static int shared_rank;
  static int foreman_rank;
  static int num_ranks;
  static int* data;
  static MPI_Comm shared_allocator_comm;
  static constexpr int N{1024};

  static void SetUpTestSuite()
  {
    auto& rm = umpire::ResourceManager::getInstance();
    auto node_allocator = rm.makeResource("SHARED"); // Defaults to MPI3 Shared Memory

    shared_allocator_comm = umpire::get_communicator_for_allocator(node_allocator, MPI_COMM_WORLD);
    MPI_Comm_size(shared_allocator_comm, &num_ranks);
    MPI_Comm_rank(shared_allocator_comm, &shared_rank);

    data = static_cast<int*>(node_allocator.allocate(N * sizeof(int)));

    if (shared_rank == foreman_rank) {
      for (int i{0}; i < N; ++i) {
        data[i] = i;
      }
    }

    MPI_Barrier(shared_allocator_comm);
  }
};

int MPISharedMemoryTest::shared_rank{0};
int MPISharedMemoryTest::foreman_rank{0};
int MPISharedMemoryTest::num_ranks{0};
int* MPISharedMemoryTest::data{nullptr};
MPI_Comm MPISharedMemoryTest::shared_allocator_comm{MPI_COMM_NULL};

TEST_F(MPISharedMemoryTest, SharedMemoryAllocation)
{
  ASSERT_NE(data, nullptr);
}

TEST_F(MPISharedMemoryTest, SharedMemoryAccess)
{
  // All processes verify the shared memory contents
  for (int i = 0; i < N; ++i) {
    ASSERT_EQ(data[i], i);
  }
}

TEST_F(MPISharedMemoryTest, SharedMemoryModification)
{
  if (shared_rank == foreman_rank) {
    data[0] = 42;
  }

  // Synchronize all processes
  MPI_Barrier(shared_allocator_comm);

  // All processes verify the modification
  ASSERT_EQ(data[0], 42);
}

TEST_F(MPISharedMemoryTest, SharedMemoryVisibility)
{
  data[shared_rank] = shared_rank * 10;

  // Synchronize all processes
  MPI_Barrier(shared_allocator_comm);

  // Verify the modifications
  for (int i = 0; i < num_ranks; ++i) {
    ASSERT_EQ(data[i], i * 10);
  }
  MPI_Barrier(shared_allocator_comm);
}

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  result = RUN_ALL_TESTS();

  umpire::cleanup_cached_communicators();

  MPI_Finalize();

  return result;
}
