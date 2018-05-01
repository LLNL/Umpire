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

#include "umpire/umpire.h"

TEST(Allocator, HostAllocatorExplicitInit)
{
  umpire_resourcemanager* rm = umpire_resourcemanager_getinstance();

  umpire_resourcemanager_initialize(rm);

  umpire_allocator* allocator = umpire_resourcemanager_get_allocator_0(rm, "HOST");

  double* test_alloc = (double*) umpire_allocator_allocate(allocator, 100*sizeof(double));

  ASSERT_NE(nullptr, test_alloc);
  umpire_allocator_deallocate(allocator, test_alloc);
  umpire_resourcemanager_delete_allocator(allocator);
}

TEST(Allocator, HostAllocator)
{
  umpire_resourcemanager* rm = umpire_resourcemanager_getinstance();
  umpire_allocator* allocator = umpire_resourcemanager_get_allocator_0(rm, "HOST");

  double* test_alloc = (double*) umpire_allocator_allocate(allocator, 100*sizeof(double));

  ASSERT_NE(nullptr, test_alloc);
  umpire_allocator_deallocate(allocator, test_alloc);
  umpire_resourcemanager_delete_allocator(allocator);
}

TEST(Allocator, HostAllocatorId)
{
  umpire_resourcemanager* rm = umpire_resourcemanager_getinstance();
  umpire_allocator* allocator = umpire_resourcemanager_get_allocator_0(rm, "HOST");
  int alloc_id = umpire_allocator_get_id(allocator);

  umpire_allocator* allocator2 = umpire_resourcemanager_get_allocator_1(rm, alloc_id);

  ASSERT_EQ(alloc_id, umpire_allocator_get_id(allocator2));

  double* test_alloc = (double*) umpire_allocator_allocate(allocator2, 100*sizeof(double));
  ASSERT_NE(nullptr, test_alloc);

  umpire_allocator_deallocate(allocator2, test_alloc);
  umpire_resourcemanager_delete_allocator(allocator);
  umpire_resourcemanager_delete_allocator(allocator2);
}

TEST(Allocator, HostAllocatorSize)
{
  umpire_resourcemanager* rm = umpire_resourcemanager_getinstance();
  umpire_allocator* allocator = umpire_resourcemanager_get_allocator_0(rm, "HOST");

  double* test_alloc = (double*) umpire_allocator_allocate(allocator, 100*sizeof(double));
  ASSERT_EQ((100*sizeof(double)), umpire_resourcemanager_get_size(rm, test_alloc));
  ASSERT_EQ((100*sizeof(double)), umpire_allocator_get_size(allocator, test_alloc));

  ASSERT_NE(nullptr, test_alloc);
  umpire_allocator_deallocate(allocator, test_alloc);
  umpire_resourcemanager_delete_allocator(allocator);
}

TEST(Allocator, HostAllocator_CurrentSize_and_HiWatermark)
{
  umpire_resourcemanager* rm = umpire_resourcemanager_getinstance();
  umpire_allocator* allocator = umpire_resourcemanager_get_allocator_0(rm, "HOST");

  double* test_alloc1 = (double*) umpire_allocator_allocate(allocator, 100*sizeof(double));
  double* test_alloc2 = (double*) umpire_allocator_allocate(allocator, 100*sizeof(double));
  double* test_alloc3 = (double*) umpire_allocator_allocate(allocator, 100*sizeof(double));

  ASSERT_EQ((100*sizeof(double)), umpire_resourcemanager_get_size(rm, test_alloc1));
  ASSERT_NE(nullptr, test_alloc1);

  ASSERT_EQ((100*sizeof(double)), umpire_allocator_get_size(allocator, test_alloc2));
  ASSERT_NE(nullptr, test_alloc2);

  ASSERT_EQ((100*sizeof(double)), umpire_allocator_get_size(allocator, test_alloc3));
  ASSERT_NE(nullptr, test_alloc3);

  ASSERT_EQ((3*100*sizeof(double)), umpire_allocator_get_current_size(allocator));
  ASSERT_EQ((3*100*sizeof(double)), umpire_allocator_get_high_watermark(allocator));

  umpire_allocator_deallocate(allocator, test_alloc1);
  ASSERT_EQ((2*100*sizeof(double)), umpire_allocator_get_current_size(allocator));
  ASSERT_EQ((3*100*sizeof(double)), umpire_allocator_get_high_watermark(allocator));

  umpire_allocator_deallocate(allocator, test_alloc2);
  ASSERT_EQ((100*sizeof(double)), umpire_allocator_get_current_size(allocator));
  ASSERT_EQ((3*100*sizeof(double)), umpire_allocator_get_high_watermark(allocator));

  umpire_allocator_deallocate(allocator, test_alloc3);
  ASSERT_EQ((0*sizeof(double)), umpire_allocator_get_current_size(allocator));
  ASSERT_EQ((3*100*sizeof(double)), umpire_allocator_get_high_watermark(allocator));

  umpire_resourcemanager_delete_allocator(allocator);
}

#if defined(UMPIRE_ENABLE_CUDA)
TEST(Allocator, DeviceAllocatorExplicitInit)
{
  umpire_resourcemanager* rm = umpire_resourcemanager_getinstance();

  umpire_resourcemanager_initialize(rm);

  umpire_allocator* allocator = umpire_resourcemanager_get_allocator_0(rm, "DEVICE");

  double* test_alloc = (double*) umpire_allocator_allocate(allocator, 100*sizeof(double));

  ASSERT_NE(nullptr, test_alloc);
  umpire_allocator_deallocate(allocator, test_alloc);
  umpire_resourcemanager_delete_allocator(allocator);
}

TEST(Allocator, DeviceAllocator)
{
  umpire_resourcemanager* rm = umpire_resourcemanager_getinstance();
  umpire_allocator* allocator = umpire_resourcemanager_get_allocator_0(rm, "DEVICE");

  double* test_alloc = (double*) umpire_allocator_allocate(allocator, 100*sizeof(double));

  ASSERT_NE(nullptr, test_alloc);
  umpire_allocator_deallocate(allocator, test_alloc);
  umpire_resourcemanager_delete_allocator(allocator);
}

TEST(Allocator, DeviceAllocatorSize)
{
  umpire_resourcemanager* rm = umpire_resourcemanager_getinstance();
  umpire_allocator* allocator = umpire_resourcemanager_get_allocator_0(rm, "DEVICE");

  double* test_alloc = (double*) umpire_allocator_allocate(allocator, 100*sizeof(double));
  ASSERT_EQ((100*sizeof(double)), umpire_resourcemanager_get_size(rm, test_alloc));
  ASSERT_EQ((100*sizeof(double)), umpire_allocator_get_size(allocator, test_alloc));

  ASSERT_NE(nullptr, test_alloc);
  umpire_allocator_deallocate(allocator, test_alloc);
  umpire_resourcemanager_delete_allocator(allocator);
}

TEST(Allocator, DeviceAllocator_CurrentSize_and_HiWatermark)
{
  umpire_resourcemanager* rm = umpire_resourcemanager_getinstance();
  umpire_allocator* allocator = umpire_resourcemanager_get_allocator_0(rm, "DEVICE");

  double* test_alloc1 = (double*) umpire_allocator_allocate(allocator, 100*sizeof(double));
  double* test_alloc2 = (double*) umpire_allocator_allocate(allocator, 100*sizeof(double));
  double* test_alloc3 = (double*) umpire_allocator_allocate(allocator, 100*sizeof(double));

  ASSERT_EQ((100*sizeof(double)), umpire_resourcemanager_get_size(rm, test_alloc1));
  ASSERT_NE(nullptr, test_alloc1);

  ASSERT_EQ((100*sizeof(double)), umpire_allocator_get_size(allocator, test_alloc2));
  ASSERT_NE(nullptr, test_alloc2);

  ASSERT_EQ((100*sizeof(double)), umpire_allocator_get_size(allocator, test_alloc3));
  ASSERT_NE(nullptr, test_alloc3);

  ASSERT_EQ((3*100*sizeof(double)), umpire_allocator_get_current_size(allocator));
  ASSERT_EQ((3*100*sizeof(double)), umpire_allocator_get_high_watermark(allocator));

  umpire_allocator_deallocate(allocator, test_alloc1);
  ASSERT_EQ((2*100*sizeof(double)), umpire_allocator_get_current_size(allocator));
  ASSERT_EQ((3*100*sizeof(double)), umpire_allocator_get_high_watermark(allocator));

  umpire_allocator_deallocate(allocator, test_alloc2);
  ASSERT_EQ((100*sizeof(double)), umpire_allocator_get_current_size(allocator));
  ASSERT_EQ((3*100*sizeof(double)), umpire_allocator_get_high_watermark(allocator));

  umpire_allocator_deallocate(allocator, test_alloc3);
  ASSERT_EQ((0*sizeof(double)), umpire_allocator_get_current_size(allocator));
  ASSERT_EQ((3*100*sizeof(double)), umpire_allocator_get_high_watermark(allocator));

  umpire_resourcemanager_delete_allocator(allocator);
}
#endif

TEST(Allocator, Deallocate)
{
  umpire_resourcemanager* rm = umpire_resourcemanager_getinstance();
  umpire_allocator* allocator = umpire_resourcemanager_get_allocator_0(rm, "HOST");
  double* test_alloc = (double*) umpire_allocator_allocate(allocator, 100*sizeof(double));
  umpire_resourcemanager_deallocate(rm, test_alloc);
  umpire_resourcemanager_delete_allocator(allocator);
  SUCCEED();
}

TEST(Allocator, DeallocateThrow)
{
  umpire_resourcemanager* rm = umpire_resourcemanager_getinstance();
  double* ptr = new double[20];
  ASSERT_ANY_THROW( umpire_resourcemanager_deallocate(rm, ptr) );

  delete[] ptr;
}

