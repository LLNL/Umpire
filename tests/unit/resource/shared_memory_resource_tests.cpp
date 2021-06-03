//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/config.hpp"
#include "gtest/gtest.h"

#include "mpi.h"

#include <algorithm>
#include <random>
#include <iostream>
#include <string>
#include <sstream>
#include <utility>
#include <vector>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/config.hpp"
#include "umpire/resource/HostSharedMemoryResource.hpp"
#include "umpire/util/MemoryResourceTraits.hpp"

namespace {
  struct SharedMemoryState {
    std::size_t largest_allocation_size;
    std::size_t initial_size;
    std::size_t num_allocations;
    std::size_t allocation_sizes[1];
  };
  const std::string shmem_state_name{"SharedMemoryState"};
  SharedMemoryState* shmem_state{nullptr};

  umpire::Allocator allocator;
  umpire::resource::HostSharedMemoryResource* shmem_resource{nullptr};

  class SharedMemoryTest : public ::testing::Test {
  public:
    using ArrayElement = int;

    const std::size_t m_segment_size{  512ULL * 1024ULL * 1024ULL };
    const std::size_t m_max_allocs{ 1024 };
    const std::size_t m_max_elems{ m_segment_size / (m_max_allocs*sizeof(ArrayElement)) };
    int m_rank;

    virtual void SetUp()
    {
      MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);

      if (shmem_resource == nullptr) {
        auto& rm = umpire::ResourceManager::getInstance();
        auto traits{umpire::get_default_resource_traits("SHARED")};
        ASSERT_EQ(traits.scope, umpire::MemoryResourceTraits::shared_scope::node);
        ASSERT_EQ(traits.resource, umpire::MemoryResourceTraits::resource_type::shared);

        traits.size = m_segment_size;
        ASSERT_NO_THROW( allocator = rm.makeResource("SHARED::node_allocator", traits); );
        auto base_strategy = allocator.getAllocationStrategy();
        shmem_resource = dynamic_cast<umpire::resource::HostSharedMemoryResource*>(base_strategy);
        MPI_Barrier(MPI_COMM_WORLD);
      }

      std::size_t state_size{ sizeof(SharedMemoryState) + (m_max_elems * sizeof(std::size_t)) };
      ASSERT_NO_THROW( shmem_state = static_cast<SharedMemoryState*>(allocator.allocate(shmem_state_name, state_size)); );

      if (m_rank == 0) {
        shmem_state->initial_size = shmem_resource->getActualSize();
        std::cout << "Initialial size is: " << shmem_state->initial_size << std::endl;
        shmem_state->largest_allocation_size = find_largest_allocation_size();

        std::random_device rd;
        std::mt19937 gen{ rd() };
        std::uniform_int_distribution<std::size_t> distrib(1, m_max_elems);

        for ( std::size_t i = 0; i < m_max_allocs; ++i) {
          shmem_state->allocation_sizes[i] = distrib(gen) * sizeof(ArrayElement);
        }

        std::vector<ArrayElement*> allocs{ alloc_until_full() };
        shmem_state->num_allocations = allocs.size();
        do_deallocations(allocs);
      }

      MPI_Barrier(MPI_COMM_WORLD);
      ASSERT_EQ(shmem_resource->getActualSize(), shmem_state->initial_size);
      MPI_Barrier(MPI_COMM_WORLD);
    }

    virtual void TearDown()
    {
      MPI_Barrier(MPI_COMM_WORLD);
      ASSERT_EQ( shmem_state->initial_size, shmem_resource->getActualSize() );

      MPI_Barrier(MPI_COMM_WORLD);

      if (m_rank == 0) {
        ASSERT_EQ( shmem_state->largest_allocation_size, find_largest_allocation_size() );
      }

      MPI_Barrier(MPI_COMM_WORLD);

      allocator.deallocate(shmem_state);
      shmem_state = nullptr;

      MPI_Barrier(MPI_COMM_WORLD);
    }

    std::size_t find_largest_allocation_size()
    {
      //
      // The shared memory allocator will use its segment for both the data and
      // supporting metadata (segment header, storage for name, block descriptor
      // linked list, etc).  Consequently, the size used to create the allocator
      // will be too large.
      //
      // This test will continue allocating until it finds the largest size.  As
      // a unit test, this test will then (de)allocate that amount in a loop to
      // insure that there are no problems within the internal accounting in the
      // implementation.
      //
      std::size_t allocation_size{m_segment_size - shmem_state->initial_size};
      void* ptr{nullptr};

      while (allocation_size != 0) {
        try {
          ptr = allocator.allocate("AllocLargest", allocation_size);
          break;
        }
        catch ( ... ) {
          allocation_size--;
          continue;
        }
      }

      allocator.deallocate(ptr);

      return allocation_size;
    }

    std::vector<ArrayElement*> alloc_until_full()
    {
      std::vector<ArrayElement*> allocs;

      try {
        for ( std::size_t i = 0; i < m_max_allocs; ++i) {
          std::stringstream name;
          name << "size_" << i;
          void* ptr{ allocator.allocate(name.str(), shmem_state->allocation_sizes[i]) };
          allocs.push_back( static_cast<ArrayElement*>(ptr) );
        } 
      } catch (...) {
        ;
      }

      return allocs;
    }

    void do_deallocations(std::vector<ArrayElement*>& allocs)
    {
      for ( auto& x : allocs ) {
        ASSERT_NO_THROW( allocator.deallocate(x); );
      }
    }

  };

  TEST_F(SharedMemoryTest, UnitTests)
  {
    MPI_Barrier(MPI_COMM_WORLD);
    {
      std::size_t allocation_size{m_segment_size+1};
      ASSERT_THROW( allocator.allocate("AllocTooMuch", allocation_size), umpire::util::Exception);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    {
      if (m_rank == 0) {
        for ( int loop{0}; loop < 100; loop++ ) {
          ASSERT_NO_THROW(
            allocator.deallocate(allocator.allocate("AllocLargest",
                                  shmem_state->largest_allocation_size)); );
        }
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    {
      auto allocs = alloc_until_full();
      int size;

      ASSERT_EQ( allocs.size(), m_max_allocs );
      ASSERT_GT(shmem_resource->getActualSize(), shmem_state->initial_size);

      MPI_Comm_size(MPI_COMM_WORLD, &size);
      for ( std::size_t i{0}; i < allocs.size(); i++ ) {
        if ( i % size != m_rank )
          continue;

        ArrayElement* buffer{ allocs[i] };
        std::size_t elems{ shmem_state->allocation_sizes[i] / sizeof(ArrayElement) };

        buffer[0] = i+1;
        buffer[elems-1] = i+1;
      }

      MPI_Barrier(MPI_COMM_WORLD);
      for ( std::size_t i{0}; i < allocs.size(); i++ ) {
        ArrayElement* buffer{ allocs[i] };
        std::size_t elems{ shmem_state->allocation_sizes[i] / sizeof(ArrayElement) };

        ASSERT_EQ( buffer[0], i+1 );
        ASSERT_EQ( buffer[elems-1], i+1 );
      }
      MPI_Barrier(MPI_COMM_WORLD);

      std::random_shuffle(allocs.begin(), allocs.end());
      do_deallocations(allocs);

      MPI_Barrier(MPI_COMM_WORLD);
      ASSERT_EQ(shmem_resource->getActualSize(), shmem_state->initial_size);

      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
}

int main(int argc, char * argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}

