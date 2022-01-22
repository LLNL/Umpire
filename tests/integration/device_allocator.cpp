//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"
#include "umpire/DeviceAllocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/device_allocator_helper.hpp"

__global__ void tester(double** data_ptr, const char* name)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx == 0) {
    umpire::DeviceAllocator da = umpire::get_device_allocator(name);
    double* data = static_cast<double*>(da.allocate(1 * sizeof(double)));
    *data_ptr = data;
    data[0] = 42 * 42;
  }
}

class DeviceAllocator : public ::testing::TestWithParam<const char*> {
/*  void TearDown() override
  {
    ASSERT_NO_THROW(umpire::destroy_device_allocator());
    ASSERT_EQ(umpire::UMPIRE_DEV_ALLOCS_h, nullptr);
  }*/
};

TEST_P(DeviceAllocator, CreateAndAllocate)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  size_t size = 1 * sizeof(double);

  umpire::DeviceAllocator da = umpire::make_device_allocator(allocator, size, GetParam());
  ASSERT_THROW((umpire::make_device_allocator(allocator, 0, "bad_da")), umpire::util::Exception);

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
  tester<<<1, 16>>>(data_ptr, GetParam());
  cudaDeviceSynchronize();
  ASSERT_EQ(*data_ptr[0], (double)(42 * 42));
}

const char* device_allocator_names [3] = {"da1", "da2", "da3"};

INSTANTIATE_TEST_SUITE_P(DeviceAllocatorTests, DeviceAllocator, ::testing::ValuesIn(device_allocator_names));

/*
void launch_kernel(double** data_ptr, const char* name)
{

}
*/
/*
  ASSERT_EQ(da1.getID(), -1);
  ASSERT_EQ(da2.getID(), -2);
  ASSERT_EQ(da3.getID(), -3);
  //umpire::DeviceAllocator da2 = umpire::make_device_allocator(allocator, size, "da2");
  //umpire::DeviceAllocator da3 = umpire::make_device_allocator(allocator, size, "da3");

  for (int i = 0; i < 3; i++) {
#if defined(UMPIRE_ENABLE_CUDA)
    ASSERT_NO_THROW(umpire::UMPIRE_DEV_ALLOCS_h[i].reset());
#elif defined(UMPIRE_ENABLE_HIP)
    hipLaunchKernelGGL(tester, dim3(1), dim3(16), 0, 0, data_ptr, name);
    hipDeviceSynchronize();
    ASSERT_EQ(*data_ptr[0], (double)(42 * 42));
#else
    FAIL(); // If neither CUDA nor HIP is enabled, yet we are testing the DeviceAllocator, something is wrong!
#endif
  }

}
*/
