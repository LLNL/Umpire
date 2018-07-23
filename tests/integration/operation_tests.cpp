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

#include "umpire/config.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/util/Exception.hpp"

class OperationTest : 
  public ::testing::TestWithParam< ::testing::tuple<std::string, std::string> >
{
  public:
    virtual void SetUp() {
      auto& rm = umpire::ResourceManager::getInstance();
      source_allocator = new umpire::Allocator(rm.getAllocator(::testing::get<0>(GetParam())));
      dest_allocator = new umpire::Allocator(rm.getAllocator(::testing::get<1>(GetParam())));

      source_array = static_cast<float*>(source_allocator->allocate(m_size*sizeof(float)));
      dest_array = static_cast<float*>(dest_allocator->allocate(m_size*sizeof(float)));

      check_array = static_cast<float*>(
          rm.getAllocator("HOST").allocate(m_size*sizeof(float)));
    }

    virtual void TearDown() {
      auto& rm = umpire::ResourceManager::getInstance();

      source_allocator->deallocate(source_array);
      dest_allocator->deallocate(dest_array);

      rm.getAllocator("HOST").deallocate(check_array);
    }

    float* source_array;
    float* dest_array;
    float* check_array;

    const size_t m_size = 1024;

    umpire::Allocator* source_allocator;
    umpire::Allocator* dest_allocator;
};

class CopyTest :
  public OperationTest
{
  public:
  void doTest() {
    auto& rm = umpire::ResourceManager::getInstance();

    for (size_t i = 0; i < m_size; i++) {
      source_array[i] = i;
    }

    rm.copy(dest_array, source_array);

    rm.copy(check_array, dest_array);

    for (size_t i = 0; i < m_size; i++) {
      ASSERT_FLOAT_EQ(source_array[i], check_array[i]);
    }
  }
};

TEST_P(CopyTest, Copy) {
  doTest();
}

#if defined(UMPIRE_ENABLE_CUDA)
INSTANTIATE_TEST_CASE_P(
    Copies,
    CopyTest,
    ::testing::Combine(
      ::testing::Values("HOST", "UM"),
      ::testing::Values("HOST", "DEVICE", "UM")
      ));
#endif

class MemsetTest :
  public OperationTest
{
  public:
  void doTest() {
    auto& rm = umpire::ResourceManager::getInstance();

    rm.memset(source_array, 0);

    rm.copy(check_array, source_array);

    for (size_t i = 0; i < m_size; i++) {
      ASSERT_EQ(0, check_array[i]);
    }
  }
};

TEST_P(MemsetTest, Memset) {
  doTest();
}

#if defined(UMPIRE_ENABLE_CUDA)
INSTANTIATE_TEST_CASE_P(
    Set,
    MemsetTest,
    ::testing::Combine(
      ::testing::Values("HOST", "UM", "DEVICE"),
      // dest allocator unecessary
      ::testing::Values("HOST")
      ));
#endif

TEST(Operation, HostToHostCopy)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  umpire::Allocator host_allocator = rm.getAllocator("HOST");

  double* array_one = static_cast<double*>(host_allocator.allocate(100*sizeof(double)));
  double* array_two = static_cast<double*>(host_allocator.allocate(100*sizeof(double)));

  for (int i = 0; i < 100; i++) {
    array_one[i] = i;
  }

  rm.copy(array_two, array_one);

  for (int i = 0; i < 100; i++) {
    ASSERT_DOUBLE_EQ(array_one[i], array_two[i]);
  }
}

#if defined(UMPIRE_ENABLE_CUDA)
TEST(Operation, HostToDeviceToHostCopy)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  umpire::Allocator host_allocator = rm.getAllocator("HOST");
  umpire::Allocator device_allocator = rm.getAllocator("DEVICE");

  double* array_one = static_cast<double*>(host_allocator.allocate(100*sizeof(double)));
  double* array_two = static_cast<double*>(device_allocator.allocate(100*sizeof(double)));

  double* array_three = static_cast<double*>(host_allocator.allocate(100*sizeof(double)));

  for (int i = 0; i < 100; i++) {
    array_one[i] = i;
  }

  rm.copy(array_two, array_one);
  rm.copy(array_three, array_two);

  for (int i = 0; i < 100; i++) {
    ASSERT_DOUBLE_EQ(array_one[i], array_three[i]);
  }
}

TEST(Operation, HostToUmCopy)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  umpire::Allocator host_allocator = rm.getAllocator("HOST");
  umpire::Allocator um_allocator = rm.getAllocator("UM");

  double* array_one = static_cast<double*>(host_allocator.allocate(100*sizeof(double)));
  double* array_two = static_cast<double*>(um_allocator.allocate(100*sizeof(double)));

  for (int i = 0; i < 100; i++) {
    array_one[i] = i;
  }

  rm.copy(array_two, array_one);

  for (int i = 0; i < 100; i++) {
    ASSERT_DOUBLE_EQ(array_one[i], array_two[i]);
  }
}

TEST(Operation, UmToHostCopy)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  umpire::Allocator host_allocator = rm.getAllocator("HOST");
  umpire::Allocator um_allocator = rm.getAllocator("UM");

  double* array_one = static_cast<double*>(um_allocator.allocate(100*sizeof(double)));
  double* array_two = static_cast<double*>(host_allocator.allocate(100*sizeof(double)));

  for (int i = 0; i < 100; i++) {
    array_one[i] = i;
  }

  rm.copy(array_two, array_one);

  for (int i = 0; i < 100; i++) {
    ASSERT_DOUBLE_EQ(array_one[i], array_two[i]);
  }
}

TEST(Operation, UmToUmCopy)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  umpire::Allocator um_allocator = rm.getAllocator("UM");

  double* array_one = static_cast<double*>(um_allocator.allocate(100*sizeof(double)));
  double* array_two = static_cast<double*>(um_allocator.allocate(100*sizeof(double)));

  for (int i = 0; i < 100; i++) {
    array_one[i] = i;
  }

  rm.copy(array_two, array_one);

  for (int i = 0; i < 100; i++) {
    ASSERT_DOUBLE_EQ(array_one[i], array_two[i]);
  }
}

TEST(Operation, UmToDeviceToUmCopy)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  umpire::Allocator um_allocator = rm.getAllocator("UM");
  umpire::Allocator device_allocator = rm.getAllocator("DEVICE");

  double* array_one = static_cast<double*>(um_allocator.allocate(100*sizeof(double)));
  double* array_two = static_cast<double*>(device_allocator.allocate(100*sizeof(double)));
  double* array_three = static_cast<double*>(um_allocator.allocate(100*sizeof(double)));

  for (int i = 0; i < 100; i++) {
    array_one[i] = i;
  }

  rm.copy(array_two, array_one);
  rm.copy(array_three, array_two);

  for (int i = 0; i < 100; i++) {
    ASSERT_DOUBLE_EQ(array_one[i], array_three[i]);
  }
}
#endif

TEST(Operation, SizeError)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  umpire::Allocator host_allocator = rm.getAllocator("HOST");

  double* array_one = static_cast<double*>(host_allocator.allocate(100*sizeof(double)));
  double* array_two = static_cast<double*>(host_allocator.allocate(70*sizeof(double)));

  ASSERT_THROW(rm.copy(array_two, array_one), umpire::util::Exception);
}

TEST(Operation, CopyOffset)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator host_allocator = rm.getAllocator("HOST");

  double* array_one = static_cast<double*>(host_allocator.allocate(100*sizeof(double)));
  double* array_two = static_cast<double*>(host_allocator.allocate(70*sizeof(double)));

  array_one[10] = 3.14;
  array_two[11] = 0.0;

  rm.copy(&array_two[11], &array_one[10], sizeof(double));

  ASSERT_EQ(array_one[10], array_two[11]);
}


TEST(Operation, HostMemset)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator host_allocator = rm.getAllocator("HOST");

  double* array = static_cast<double*>(host_allocator.allocate(100*sizeof(double)));

  rm.memset(array, 0);

  for (int i = 0; i < 100; i++) {
    ASSERT_DOUBLE_EQ(array[i], 0);
  }
}

#if defined(UMPIRE_ENABLE_CUDA)
TEST(Operation, DeviceMemset)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  umpire::Allocator host_allocator = rm.getAllocator("HOST");
  umpire::Allocator device_allocator = rm.getAllocator("DEVICE");

  double* h_array = static_cast<double*>(host_allocator.allocate(100*sizeof(double)));
  double* d_array = static_cast<double*>(device_allocator.allocate(100*sizeof(double)));

  rm.memset(d_array, 0);

  rm.copy(h_array, d_array);

  for (int i = 0; i < 100; i++) {
    ASSERT_DOUBLE_EQ(h_array[i], 0);
  }
}

TEST(Operation, UmMemset)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator host_allocator = rm.getAllocator("UM");

  double* array = static_cast<double*>(host_allocator.allocate(100*sizeof(double)));

  rm.memset(array, 0);

  for (int i = 0; i < 100; i++) {
    ASSERT_DOUBLE_EQ(array[i], 0);
  }
}
#endif

TEST(Operation, HostReallocate)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator host_allocator = rm.getAllocator("HOST");

  double* array = static_cast<double*>(host_allocator.allocate(100*sizeof(double)));

  for (int i = 0; i < 100; i++) {
    array[i] = static_cast<double>(i);
  }

  double* new_array = static_cast<double*>(rm.reallocate(array, 50*sizeof(double)));

  ASSERT_EQ(host_allocator.getSize(new_array), 50*sizeof(double));


  for (int i = 0; i < 50; i++) {
    ASSERT_DOUBLE_EQ(new_array[i], static_cast<double>(i));
  }
}

TEST(Operation, HostReallocateLarger)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator host_allocator = rm.getAllocator("HOST");

  double* array = static_cast<double*>(host_allocator.allocate(100*sizeof(double)));

  for (int i = 0; i < 100; i++) {
    array[i] = static_cast<double>(i);
  }

  double* new_array = static_cast<double*>(rm.reallocate(array, 150*sizeof(double)));

  ASSERT_EQ(host_allocator.getSize(new_array), 150*sizeof(double));

  for (int i = 0; i < 100; i++) {
    ASSERT_DOUBLE_EQ(new_array[i], static_cast<double>(i));
  }
}

#if defined(UMPIRE_ENABLE_CUDA)
TEST(Operation, GenericReallocateDevice)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  umpire::Allocator device_allocator = rm.getAllocator("DEVICE");
  umpire::Allocator host_allocator = rm.getAllocator("HOST");

  double* d_array = static_cast<double*>(device_allocator.allocate(100*sizeof(double)));
  rm.memset(d_array, 0);

  double* d_new_array = static_cast<double*>(rm.reallocate(d_array, 50*sizeof(double)));

  double* h_array = static_cast<double*>(host_allocator.allocate(50*sizeof(double)));

  ASSERT_EQ(device_allocator.getSize(d_new_array), 50*sizeof(double));

  rm.copy(h_array, d_new_array);

  for (int i = 0; i < 50; i++) {
    ASSERT_DOUBLE_EQ(h_array[i], 0);
  }
}

TEST(Operation, GenericReallocateLarger)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  umpire::Allocator device_allocator = rm.getAllocator("DEVICE");
  umpire::Allocator host_allocator = rm.getAllocator("HOST");

  double* d_array = static_cast<double*>(device_allocator.allocate(100*sizeof(double)));

  rm.memset(d_array, 0);

  double* d_new_array = static_cast<double*>(rm.reallocate(d_array, 150*sizeof(double)));

  double* h_array = static_cast<double*>(host_allocator.allocate(150*sizeof(double)));

  ASSERT_EQ(device_allocator.getSize(d_new_array), 150*sizeof(double));

  rm.copy(h_array, d_new_array);

  for (int i = 0; i < 100; i++) {
    ASSERT_DOUBLE_EQ(h_array[i], 0);
  }
}
#endif

TEST(Operation, MoveHostToHost)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  umpire::Allocator host_allocator = rm.getAllocator("HOST");
  double* host_array = static_cast<double*>(host_allocator.allocate(100*sizeof(double)));

  for (int i = 0; i < 100; i++) {
    host_array[i] = i;
  }

  double* moved_array = static_cast<double*>(rm.move(host_array, host_allocator));

  ASSERT_EQ(moved_array, host_array);

  for (int i = 0; i < 100; i++) {
    ASSERT_DOUBLE_EQ(moved_array[i], i);
  }
}

#if defined(UMPIRE_ENABLE_CUDA)
TEST(Operation, MoveHostToDeviceToHost)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  umpire::Allocator host_allocator = rm.getAllocator("HOST");
  umpire::Allocator device_allocator = rm.getAllocator("DEVICE");

  double* h_array = static_cast<double*>(host_allocator.allocate(100*sizeof(double)));

  for (int i = 0; i < 100; i++) {
    h_array[i] = i;
  }

  double* moved_array_d = static_cast<double*>(rm.move(h_array, device_allocator));

  double* moved_array = static_cast<double*>(rm.move(moved_array_d, host_allocator));

  for (int i = 0; i < 100; i++) {
    ASSERT_DOUBLE_EQ(moved_array[i], i);
  }
}

TEST(Operation, MoveHostToUM)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  umpire::Allocator host_allocator = rm.getAllocator("HOST");
  umpire::Allocator um_allocator = rm.getAllocator("UM");

  double* h_array = static_cast<double*>(host_allocator.allocate(100*sizeof(double)));

  for (int i = 0; i < 100; i++) {
    h_array[i] = i;
  }

  double* moved_array = static_cast<double*>(rm.move(h_array, um_allocator));

  for (int i = 0; i < 100; i++) {
    ASSERT_DOUBLE_EQ(moved_array[i], i);
  }
}
#endif
