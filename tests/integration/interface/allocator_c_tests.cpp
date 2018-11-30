//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"

#include "umpire/interface/umpire.h"

class AllocatorCTest :
  public ::testing::TestWithParam< const char* >
{
  public:
  virtual void SetUp()
  {
    um_resourcemanager rm;
    um_resourcemanager_get_instance(&rm);
    um_resourcemanager_get_allocator_by_name(&rm, GetParam(), &m_allocator);;
  }

  virtual void TearDown()
  {
  }

  um_allocator m_allocator;

  const size_t m_big = 64;
  const size_t m_small = 8;
  const size_t m_nothing = 0;
};

TEST_P(AllocatorCTest, AllocateDeallocateBig)
{
  double* data = (double*) um_allocator_allocate(&m_allocator, m_big*sizeof(double));
  ASSERT_NE(nullptr, data);

  um_allocator_deallocate(&m_allocator, data);
}

const char* allocator_names[] = {
  "HOST"
#if defined(UMPIRE_ENABLE_DEVICE)
  , "DEVICE"
#endif
#if defined(UMPIRE_ENABLE_UM)
  , "UM"
#endif
#if defined(UMPIRE_ENABLE_CUDA)
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

// 
// TEST(Allocator, HostAllocatorId)
// {
//   um_resourcemanager* rm = um_resourcemanager_get_instance();
//   um_allocator* allocator = um_resourcemanager_get_allocator_0(rm, "HOST");
//   int alloc_id = um_allocator_get_id(allocator);
// 
//   um_allocator* allocator2 = um_resourcemanager_get_allocator_1(rm, alloc_id);
// 
//   ASSERT_EQ(alloc_id, um_allocator_get_id(allocator2));
// 
//   double* test_alloc = (double*) um_allocator_allocate(allocator2, 100*sizeof(double));
//   ASSERT_NE(nullptr, test_alloc);
// 
//   um_allocator_deallocate(allocator2, test_alloc);
//   um_resourcemanager_delete_allocator(allocator);
//   um_resourcemanager_delete_allocator(allocator2);
// }
// 
// TEST(Allocator, HostAllocatorSize)
// {
//   um_resourcemanager* rm = um_resourcemanager_get_instance();
//   um_allocator* allocator = um_resourcemanager_get_allocator_0(rm, "HOST");
// 
//   double* test_alloc = (double*) um_allocator_allocate(allocator, 100*sizeof(double));
//   ASSERT_EQ((100*sizeof(double)), um_resourcemanager_get_size(rm, test_alloc));
//   ASSERT_EQ((100*sizeof(double)), um_allocator_get_size(allocator, test_alloc));
// 
//   ASSERT_NE(nullptr, test_alloc);
//   um_allocator_deallocate(allocator, test_alloc);
//   um_resourcemanager_delete_allocator(allocator);
// }
// 
// TEST(Allocator, HostAllocator_CurrentSize_and_HiWatermark)
// {
//   um_resourcemanager* rm = um_resourcemanager_get_instance();
//   um_allocator* allocator = um_resourcemanager_get_allocator_0(rm, "HOST");
// 
//   double* test_alloc1 = (double*) um_allocator_allocate(allocator, 100*sizeof(double));
//   double* test_alloc2 = (double*) um_allocator_allocate(allocator, 100*sizeof(double));
//   double* test_alloc3 = (double*) um_allocator_allocate(allocator, 100*sizeof(double));
// 
//   ASSERT_EQ((100*sizeof(double)), um_resourcemanager_get_size(rm, test_alloc1));
//   ASSERT_NE(nullptr, test_alloc1);
// 
//   ASSERT_EQ((100*sizeof(double)), um_allocator_get_size(allocator, test_alloc2));
//   ASSERT_NE(nullptr, test_alloc2);
// 
//   ASSERT_EQ((100*sizeof(double)), um_allocator_get_size(allocator, test_alloc3));
//   ASSERT_NE(nullptr, test_alloc3);
// 
//   ASSERT_EQ((3*100*sizeof(double)), um_allocator_get_current_size(allocator));
//   ASSERT_EQ((3*100*sizeof(double)), um_allocator_get_high_watermark(allocator));
// 
//   um_allocator_deallocate(allocator, test_alloc1);
//   ASSERT_EQ((2*100*sizeof(double)), um_allocator_get_current_size(allocator));
//   ASSERT_EQ((3*100*sizeof(double)), um_allocator_get_high_watermark(allocator));
// 
//   um_allocator_deallocate(allocator, test_alloc2);
//   ASSERT_EQ((100*sizeof(double)), um_allocator_get_current_size(allocator));
//   ASSERT_EQ((3*100*sizeof(double)), um_allocator_get_high_watermark(allocator));
// 
//   um_allocator_deallocate(allocator, test_alloc3);
//   ASSERT_EQ((0*sizeof(double)), um_allocator_get_current_size(allocator));
//   ASSERT_EQ((3*100*sizeof(double)), um_allocator_get_high_watermark(allocator));
// 
//   um_resourcemanager_delete_allocator(allocator);
// }
// 
// #if defined(um_ENABLE_CUDA)
// TEST(Allocator, DeviceAllocatorExplicitInit)
// {
//   um_resourcemanager* rm = um_resourcemanager_get_instance();
// 
//   um_resourcemanager_initialize(rm);
// 
//   um_allocator* allocator = um_resourcemanager_get_allocator_0(rm, "DEVICE");
// 
//   double* test_alloc = (double*) um_allocator_allocate(allocator, 100*sizeof(double));
// 
//   ASSERT_NE(nullptr, test_alloc);
//   um_allocator_deallocate(allocator, test_alloc);
//   um_resourcemanager_delete_allocator(allocator);
// }
// 
// TEST(Allocator, DeviceAllocator)
// {
//   um_resourcemanager* rm = um_resourcemanager_get_instance();
//   um_allocator* allocator = um_resourcemanager_get_allocator_0(rm, "DEVICE");
// 
//   double* test_alloc = (double*) um_allocator_allocate(allocator, 100*sizeof(double));
// 
//   ASSERT_NE(nullptr, test_alloc);
//   um_allocator_deallocate(allocator, test_alloc);
//   um_resourcemanager_delete_allocator(allocator);
// }
// 
// TEST(Allocator, DeviceAllocatorSize)
// {
//   um_resourcemanager* rm = um_resourcemanager_get_instance();
//   um_allocator* allocator = um_resourcemanager_get_allocator_0(rm, "DEVICE");
// 
//   double* test_alloc = (double*) um_allocator_allocate(allocator, 100*sizeof(double));
//   ASSERT_EQ((100*sizeof(double)), um_resourcemanager_get_size(rm, test_alloc));
//   ASSERT_EQ((100*sizeof(double)), um_allocator_get_size(allocator, test_alloc));
// 
//   ASSERT_NE(nullptr, test_alloc);
//   um_allocator_deallocate(allocator, test_alloc);
//   um_resourcemanager_delete_allocator(allocator);
// }
// 
// TEST(Allocator, DeviceAllocator_CurrentSize_and_HiWatermark)
// {
//   um_resourcemanager* rm = um_resourcemanager_get_instance();
//   um_allocator* allocator = um_resourcemanager_get_allocator_0(rm, "DEVICE");
// 
//   double* test_alloc1 = (double*) um_allocator_allocate(allocator, 100*sizeof(double));
//   double* test_alloc2 = (double*) um_allocator_allocate(allocator, 100*sizeof(double));
//   double* test_alloc3 = (double*) um_allocator_allocate(allocator, 100*sizeof(double));
// 
//   ASSERT_EQ((100*sizeof(double)), um_resourcemanager_get_size(rm, test_alloc1));
//   ASSERT_NE(nullptr, test_alloc1);
// 
//   ASSERT_EQ((100*sizeof(double)), um_allocator_get_size(allocator, test_alloc2));
//   ASSERT_NE(nullptr, test_alloc2);
// 
//   ASSERT_EQ((100*sizeof(double)), um_allocator_get_size(allocator, test_alloc3));
//   ASSERT_NE(nullptr, test_alloc3);
// 
//   ASSERT_EQ((3*100*sizeof(double)), um_allocator_get_current_size(allocator));
//   ASSERT_EQ((3*100*sizeof(double)), um_allocator_get_high_watermark(allocator));
// 
//   um_allocator_deallocate(allocator, test_alloc1);
//   ASSERT_EQ((2*100*sizeof(double)), um_allocator_get_current_size(allocator));
//   ASSERT_EQ((3*100*sizeof(double)), um_allocator_get_high_watermark(allocator));
// 
//   um_allocator_deallocate(allocator, test_alloc2);
//   ASSERT_EQ((100*sizeof(double)), um_allocator_get_current_size(allocator));
//   ASSERT_EQ((3*100*sizeof(double)), um_allocator_get_high_watermark(allocator));
// 
//   um_allocator_deallocate(allocator, test_alloc3);
//   ASSERT_EQ((0*sizeof(double)), um_allocator_get_current_size(allocator));
//   ASSERT_EQ((3*100*sizeof(double)), um_allocator_get_high_watermark(allocator));
// 
//   um_resourcemanager_delete_allocator(allocator);
// }
// #endif
// 
// TEST(Allocator, Deallocate)
// {
//   um_resourcemanager* rm = um_resourcemanager_get_instance();
//   um_allocator* allocator = um_resourcemanager_get_allocator_0(rm, "HOST");
//   double* test_alloc = (double*) um_allocator_allocate(allocator, 100*sizeof(double));
//   um_resourcemanager_deallocate(rm, test_alloc);
//   um_resourcemanager_delete_allocator(allocator);
//   SUCCEED();
// }
// 
// TEST(Allocator, DeallocateThrow)
// {
//   um_resourcemanager* rm = um_resourcemanager_get_instance();
//   double* ptr = new double[20];
//   ASSERT_ANY_THROW( um_resourcemanager_deallocate(rm, ptr) );
// 
//   delete[] ptr;
// }
// 
