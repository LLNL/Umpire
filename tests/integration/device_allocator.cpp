//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"
#include "umpire/DeviceAllocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/device_allocator_helper.hpp"

constexpr double NUM = 42.0 * 42.0;
static char* device_allocator_names_h[3] = {"da1", "da2", "da3"};
__device__ static char* device_allocator_names[3] = {"da1", "da2", "da3"};

__global__ void tester(double** data_ptr, int index)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx == 0) {
    umpire::DeviceAllocator da = umpire::get_device_allocator(device_allocator_names[index]);
    double* data = static_cast<double*>(da.allocate(1 * sizeof(double)));
    *data_ptr = data;
    data[0] = NUM;
  }
}

class DeviceAllocator : public ::testing::TestWithParam<char*> {
 public:
  static void TearDownTestSuite()
  {
    ASSERT_NO_THROW(umpire::destroy_device_allocator());
    ASSERT_EQ(umpire::UMPIRE_DEV_ALLOCS_h, nullptr);
  }
};

TEST_P(DeviceAllocator, CreateAndAllocate)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  size_t size = 1 * sizeof(double);

  ASSERT_FALSE(umpire::is_device_allocator("No_DeviceAllocator_created_yet"));

  umpire::DeviceAllocator da = umpire::make_device_allocator(allocator, size, GetParam());
  ASSERT_THROW((umpire::make_device_allocator(allocator, 0, "bad_da")), umpire::runtime_error);

  ASSERT_TRUE(da.isInitialized());
  ASSERT_TRUE(umpire::is_device_allocator(da.getName()));
  ASSERT_TRUE(umpire::is_device_allocator(da.getID()));

  ASSERT_FALSE(umpire::is_device_allocator("not_da"));
  ASSERT_FALSE(umpire::is_device_allocator(0));

  ASSERT_NO_THROW(UMPIRE_SET_UP_DEVICE_ALLOCATORS());
}

TEST_P(DeviceAllocator, LaunchKernelTest)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");

  double** data_ptr = static_cast<double**>(allocator.allocate(sizeof(double*)));

  int str_index{0};
  for (int i = 0; i < 3; i++) {
    if (strcmp(device_allocator_names_h[i], GetParam()) == 0) {
      str_index = i;
    }
  }

#if defined(UMPIRE_ENABLE_CUDA)
  tester<<<1, 16>>>(data_ptr, str_index);
  cudaDeviceSynchronize();
#elif defined(UMPIRE_ENABLE_HIP)
  hipLaunchKernelGGL(tester, dim3(1), dim3(16), 0, 0, data_ptr, str_index);
  hipDeviceSynchronize();
#endif

  ASSERT_EQ(*data_ptr[0], NUM);

  auto my_da = umpire::get_device_allocator(GetParam());
  ASSERT_NO_THROW(my_da.reset());

  ASSERT_EQ(my_da.getCurrentSize(), 0);

#if defined(UMPIRE_ENABLE_CUDA)
  tester<<<1, 16>>>(data_ptr, str_index);
  cudaDeviceSynchronize();
#elif defined(UMPIRE_ENABLE_HIP)
  hipLaunchKernelGGL(tester, dim3(1), dim3(16), 0, 0, data_ptr, str_index);
  hipDeviceSynchronize();
#endif

  ASSERT_EQ(my_da.getCurrentSize(), sizeof(double));
}

INSTANTIATE_TEST_SUITE_P(DeviceAllocatorTests, DeviceAllocator, ::testing::ValuesIn(device_allocator_names_h));
