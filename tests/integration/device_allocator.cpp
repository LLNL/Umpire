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

TEST(DeviceAllocator, CreateAndAllocate)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  size_t size = 3 * sizeof(double);

  ASSERT_EQ(umpire::UMPIRE_DEV_ALLOCS_h, nullptr);

  umpire::DeviceAllocator da1 = umpire::make_device_allocator(allocator, size, "da1");
  umpire::DeviceAllocator da2 = umpire::make_device_allocator(allocator, size, "da2");
  umpire::DeviceAllocator da3 = umpire::make_device_allocator(allocator, size, "da3");
  ASSERT_THROW((umpire::make_device_allocator(allocator, 0, "bad_da")), umpire::util::Exception);

  ASSERT_NE(umpire::UMPIRE_DEV_ALLOCS_h, nullptr);

  ASSERT_EQ(da1.getID(), -1);
  ASSERT_EQ(da2.getID(), -2);
  ASSERT_EQ(da3.getID(), -3);
  ASSERT_EQ(da2.isInitialized(), true);

  ASSERT_EQ(umpire::is_device_allocator(da1.getName()), true);
  ASSERT_EQ(umpire::is_device_allocator(da1.getID()), true);
  ASSERT_EQ(umpire::is_device_allocator(da2.getID()), true);
  ASSERT_EQ(umpire::is_device_allocator(da3.getID()), true);
  ASSERT_EQ(umpire::is_device_allocator("not_da"), false);
  ASSERT_EQ(umpire::is_device_allocator(1), false);
  ASSERT_EQ(umpire::is_device_allocator(-5), false);

  ASSERT_NO_THROW(UMPIRE_SET_UP_DEVICE_ALLOCATORS());

  double** data_ptr = static_cast<double**>(allocator.allocate(sizeof(double*)));

  for (int i = 0; i < 3; i++) {
#if defined(UMPIRE_ENABLE_CUDA)
    tester<<<1, 16>>>(data_ptr, umpire::UMPIRE_DEV_ALLOCS_h[0].getName());
    cudaDeviceSynchronize();
    ASSERT_EQ(*data_ptr[0], (double)(42 * 42));
    ASSERT_NO_THROW(umpire::UMPIRE_DEV_ALLOCS_h[i].reset());
#elif defined(UMPIRE_ENABLE_HIP)
    hipLaunchKernelGGL(tester, dim3(1), dim3(16), 0, 0, data_ptr, name);
    hipDeviceSynchronize();
    ASSERT_EQ(*data_ptr[0], (double)(42 * 42));
#else
    FAIL(); // If neither CUDA nor HIP is enabled, yet we are testing the DeviceAllocator, something is wrong!
#endif
  }

  ASSERT_NO_THROW(umpire::destroy_device_allocator());
  ASSERT_EQ(umpire::UMPIRE_DEV_ALLOCS_h, nullptr);
  allocator.deallocate(data_ptr);
}
