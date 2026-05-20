//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "gtest/gtest.h"
#include "mpi.h"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/config.hpp"

TEST(GetCommunicator, Null)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("HOST");

  ASSERT_EQ(umpire::get_communicator_for_allocator(allocator, MPI_COMM_WORLD), MPI_COMM_NULL);
}

TEST(GetCommunicator, SharedAndCached)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto traits{umpire::get_default_resource_traits("SHARED::MPI3")};
  traits.size = 4096;

  // NOTE: The name of the allocator MUST have "SHARED::MPI3:: prefix when both IPC and MPI3 enabled.
  auto allocator = rm.makeResource("SHARED::MPI3::node_allocator", traits);

  auto comm = umpire::get_communicator_for_allocator(allocator, MPI_COMM_WORLD);
  ASSERT_NE(comm, MPI_COMM_NULL);

  auto cached_comm = umpire::get_communicator_for_allocator(allocator, MPI_COMM_WORLD);
  int result;
  MPI_Comm_compare(comm, cached_comm, &result);
  ASSERT_EQ(result, MPI_IDENT);
}

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
