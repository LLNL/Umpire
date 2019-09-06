//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"

#include "umpire/config.hpp"
#include "umpire/interface/umpire.h"

static int unique_name = 0;

class AllocatorCTest :
  public ::testing::TestWithParam< const char* >
{
  public:
  virtual void SetUp()
  {
    umpire_resourcemanager rm;
    umpire_resourcemanager_get_instance(&rm);
    umpire_resourcemanager_get_allocator_by_name(&rm, GetParam(), &m_allocator);;
  }

  virtual void TearDown()
  {
    umpire_allocator_delete(&m_allocator);
  }

  umpire_allocator m_allocator;

  const std::size_t m_big = 64;
  const std::size_t m_small = 8;
  const std::size_t m_nothing = 0;
};

TEST_P(AllocatorCTest, AllocateDeallocateBig)
{
  double* data = (double*) umpire_allocator_allocate(&m_allocator, m_big*sizeof(double));
  ASSERT_NE(nullptr, data);

  umpire_allocator_deallocate(&m_allocator, data);
}

TEST_P(AllocatorCTest, AllocateDeallocateSmall)
{
  double* data = (double*) umpire_allocator_allocate(&m_allocator, m_small*sizeof(double));
  ASSERT_NE(nullptr, data);

  umpire_allocator_deallocate(&m_allocator, data);
}

TEST_P(AllocatorCTest, AllocateDeallocateNothing)
{
  double* data = (double*) umpire_allocator_allocate(&m_allocator, m_nothing*sizeof(double));
  ASSERT_NE(nullptr, data);

  umpire_allocator_deallocate(&m_allocator, data);
}

TEST_P(AllocatorCTest, GetSize)
{
  const std::size_t size = m_big * sizeof(double);

  double* data = (double*) umpire_allocator_allocate(&m_allocator, m_big*sizeof(double));

  ASSERT_EQ(size, umpire_allocator_get_size(&m_allocator, data));

  umpire_allocator_deallocate(&m_allocator, data);
}

TEST_P(AllocatorCTest, GetAllocatorById)
{
  umpire_resourcemanager rm;
  umpire_resourcemanager_get_instance(&rm);
  int alloc_id = umpire_allocator_get_id(&m_allocator);

  umpire_allocator alloc_two;
  umpire_resourcemanager_get_allocator_by_id(&rm, alloc_id, &alloc_two);
  ASSERT_EQ(alloc_id, umpire_allocator_get_id(&alloc_two));

  umpire_allocator_delete(&alloc_two);
}

TEST_P(AllocatorCTest, SizeAndHighWatermark)
{
  double* data_one = (double*) umpire_allocator_allocate(&m_allocator, m_big*sizeof(double));
  ASSERT_NE(nullptr, data_one);

  double* data_two = (double*) umpire_allocator_allocate(&m_allocator, m_big*sizeof(double));
  ASSERT_NE(nullptr, data_two);

  double* data_three = (double*) umpire_allocator_allocate(&m_allocator, m_big*sizeof(double));
  ASSERT_NE(nullptr, data_three);

  std::size_t total_size = 3*m_big*sizeof(double);

  ASSERT_EQ(total_size, umpire_allocator_get_current_size(&m_allocator));
  ASSERT_EQ(total_size, umpire_allocator_get_high_watermark(&m_allocator));

  umpire_allocator_deallocate(&m_allocator, data_three);
  ASSERT_EQ((2*m_big*sizeof(double)), umpire_allocator_get_current_size(&m_allocator));
  ASSERT_EQ(total_size, umpire_allocator_get_high_watermark(&m_allocator));

  umpire_allocator_deallocate(&m_allocator, data_two);
  ASSERT_EQ((m_big*sizeof(double)), umpire_allocator_get_current_size(&m_allocator));
  ASSERT_EQ(total_size, umpire_allocator_get_high_watermark(&m_allocator));

  umpire_allocator_deallocate(&m_allocator, data_one);
  ASSERT_EQ(0, umpire_allocator_get_current_size(&m_allocator));
  ASSERT_EQ(total_size, umpire_allocator_get_high_watermark(&m_allocator));
}

TEST_P(AllocatorCTest, IsAllocator)
{
  umpire_resourcemanager rm;
  umpire_resourcemanager_get_instance(&rm);
  ASSERT_EQ(true, umpire_resourcemanager_is_allocator(&rm, GetParam()));
}

TEST_P(AllocatorCTest, HasAllocator)
{
  umpire_resourcemanager rm;
  umpire_resourcemanager_get_instance(&rm);

  double* data = (double*) umpire_allocator_allocate(&m_allocator, m_big*sizeof(double));
  ASSERT_NE(nullptr, data);

  ASSERT_EQ(true, umpire_resourcemanager_has_allocator(&rm, (void*)data));
  umpire_allocator_deallocate(&m_allocator, data);
  ASSERT_EQ(false, umpire_resourcemanager_has_allocator(&rm, (void*)data));
}

const char* allocator_names[] = {
  "HOST"
#if defined(UMPIRE_ENABLE_DEVICE)
  , "DEVICE"
#endif
#if defined(UMPIRE_ENABLE_UM)
  , "UM"
#endif
#if defined(UMPIRE_ENABLE_CUDA) || defined(UMPIRE_ENABLE_HIP)
  , "DEVICE_CONST"
#endif
#if defined(UMPIRE_ENABLE_PINNED)
  , "PINNED"
#endif
};

INSTANTIATE_TEST_CASE_P(
    Allocators,
    AllocatorCTest,
    ::testing::ValuesIn(allocator_names));

class PoolAllocatorCTest :
  public ::testing::TestWithParam< const char* >
{
  public:
  virtual void SetUp()
  {
    std::string pool_name = std::string{GetParam()} + "_c_pool" + std::to_string(unique_name++);

    umpire_resourcemanager rm;
    umpire_resourcemanager_get_instance(&rm);
    umpire_resourcemanager_get_allocator_by_name(&rm, GetParam(), &m_allocator);;
    umpire_resourcemanager_make_allocator_pool(
        &rm, pool_name.c_str(), m_allocator, m_big, m_small, &m_pool);
  }

  virtual void TearDown()
  {
    umpire_allocator_delete(&m_allocator);
    umpire_allocator_delete(&m_pool);
  }

  umpire_allocator m_allocator;
  umpire_allocator m_pool;

#if defined(UMPIRE_ENABLE_DEVICE)
  const std::size_t m_pool_init = 4294967296 + 64;
#else
  const std::size_t m_pool_init = 1024 * 1024 * 64;
#endif
  const std::size_t m_big = 1024 * 1024;
  const std::size_t m_small = 64;
  const std::size_t m_nothing = 0;
};

TEST_P(PoolAllocatorCTest, AllocateDeallocateBig)
{
  double* data = (double*) umpire_allocator_allocate(&m_allocator, m_big*sizeof(double));
  ASSERT_NE(nullptr, data);

  umpire_allocator_deallocate(&m_allocator, data);
}

TEST_P(PoolAllocatorCTest, AllocateDeallocateSmall)
{
  double* data = (double*) umpire_allocator_allocate(&m_allocator, m_small*sizeof(double));
  ASSERT_NE(nullptr, data);

  umpire_allocator_deallocate(&m_allocator, data);
}

TEST_P(PoolAllocatorCTest, AllocateDeallocateNothing)
{
  double* data = (double*) umpire_allocator_allocate(&m_allocator, m_nothing*sizeof(double));
  ASSERT_NE(nullptr, data);

  umpire_allocator_deallocate(&m_allocator, data);
}

const char* pool_names[] = {
  "HOST"
#if defined(UMPIRE_ENABLE_DEVICE)
  , "DEVICE"
#endif
#if defined(UMPIRE_ENABLE_UM)
  , "UM"
#endif
#if defined(UMPIRE_ENABLE_PINNED)
  , "PINNED"
#endif
};

INSTANTIATE_TEST_CASE_P(
    Pools,
    PoolAllocatorCTest,
    ::testing::ValuesIn(pool_names));

//TEST(AllocatorC, RegisterAllocator)
//{
//  umpire_resourcemanager rm;
//  umpire_resourcemanager_get_instance(&rm);
//
//  umpire_allocator alloc;
//  umpire_resourcemanager_get_allocator_by_name(&rm, "HOST", &alloc);
//
//  umpire_resourcemanager_register_allocator(
//      "my_host_allocator_copy",
//      &alloc);
//
//  SUCCEED();
//}

// TEST(Allocator, DeallocateThrow)
// {
//   umpire_resourcemanager* rm = umpire_resourcemanager_get_instance();
//   double* ptr = new double[20];
//   ASSERT_ANY_THROW( umpire_resourcemanager_deallocate(rm, ptr) );
//
//   delete[] ptr;
// }
//
