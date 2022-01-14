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

template <typename Platform>
struct allocate_and_use {
};

#if defined(UMPIRE_ENABLE_CUDA) || defined(UMPIRE_ENABLE_HIP)
__global__ void tester(double** data_ptr, const char* name)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx == 0) {
    umpire::DeviceAllocator da = umpire::get_device_allocator(name);
    double* data = static_cast<double*>(da.allocate(42 * sizeof(double)));
    *data_ptr = data;
    data[0] = 42 * 42;
  }
}
#endif

#if defined(UMPIRE_ENABLE_CUDA)
struct cuda_platform {
};

template <>
struct allocate_and_use<cuda_platform> {
  void test(umpire::Allocator alloc, const char* name)
  {
    double** data_ptr = static_cast<double**>(alloc.allocate(sizeof(double*)));
    tester<<<1, 16>>>(data_ptr, name);
    cudaDeviceSynchronize();
    ASSERT_EQ(*data_ptr[0], (double)(42 * 42));
    alloc.deallocate(data_ptr);
  }
};
#endif

#if defined(UMPIRE_ENABLE_HIP)
struct hip_platform {
};

template <>
struct allocate_and_use<hip_platform> {
  void test(umpire::Allocator alloc, const char* name)
  {
    double** data_ptr = static_cast<double**>(alloc.allocate(sizeof(double*)));
    hipLaunchKernelGGL(tester, dim3(1), dim3(16), 0, 0, data_ptr, name);
    hipDeviceSynchronize();
    ASSERT_EQ(*data_ptr[0], (double)(42 * 42));
    alloc.deallocate(data_ptr);
  }
};
#endif

TEST(DeviceAllocator, CreateAndAllocate)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  size_t size = 2048;
  // const char* name = "da";

  ASSERT_EQ(umpire::UMPIRE_DEV_ALLOCS_h, nullptr);

  umpire::DeviceAllocator dev_alloc = umpire::make_device_allocator(allocator, size, "da");
  
  ASSERT_NE(umpire::UMPIRE_DEV_ALLOCS_h, nullptr);

  ASSERT_EQ(dev_alloc.getID(), -1);
  // ASSERT_EQ(dev_alloc.getName(), name);
  ASSERT_EQ(dev_alloc.isInitialized(), true);

  ASSERT_EQ(umpire::is_device_allocator("da"), true);
  ASSERT_EQ(umpire::is_device_allocator(dev_alloc.getID()), true);
  ASSERT_EQ(umpire::is_device_allocator("not_da"), false);
  ASSERT_EQ(umpire::is_device_allocator(1), false);

  UMPIRE_SET_UP_DEVICE_ALLOCATORS();

#if defined(UMPIRE_ENABLE_CUDA)
  allocate_and_use<cuda_platform> cuda;
  ASSERT_NO_THROW(cuda.test(allocator, "da"));
#elif defined(UMPIRE_ENABLE_HIP)
  allocate_and_use<hip_platform> hip;
  ASSERT_NO_THROW(hip.test(allocator, "da"));
#else
  FAIL();   // If neither CUDA nor HIP is enabled, yet we are testing the DeviceAllocator, something is wrong!
#endif

  ASSERT_NO_THROW(dev_alloc.reset());
  ASSERT_NO_THROW(umpire::destroy_device_allocator());
}

TEST(DeviceAllocator, MultipleDAs)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  size_t size = 42 * sizeof(double);

  umpire::DeviceAllocator da1 = umpire::make_device_allocator(allocator, size, "da1");
  umpire::DeviceAllocator da2 = umpire::make_device_allocator(allocator, size, "da2");
  umpire::DeviceAllocator da3 = umpire::make_device_allocator(allocator, size, "da3");
  umpire::DeviceAllocator da4 = umpire::make_device_allocator(allocator, size, "da4");

  ASSERT_NE(umpire::UMPIRE_DEV_ALLOCS_h, nullptr);

  ASSERT_EQ(da1.getID(), -1);
  ASSERT_EQ(da2.getID(), -2);
  ASSERT_EQ(da3.getID(), -3);
  ASSERT_EQ(da4.getID(), -4);

  ASSERT_EQ(da1.getName(), "da1");
  ASSERT_EQ(da2.getName(), "da2");
  ASSERT_EQ(da3.getName(), "da3");
  ASSERT_EQ(da4.getName(), "da4");

  ASSERT_EQ(umpire::is_device_allocator("da2"), true);
  ASSERT_EQ(umpire::is_device_allocator(da3.getID()), true);
  ASSERT_EQ(umpire::is_device_allocator("not_da"), false);
  ASSERT_EQ(umpire::is_device_allocator(-5), false);

  UMPIRE_SET_UP_DEVICE_ALLOCATORS();

  for (int i = 0; i < umpire::UMPIRE_TOTAL_DEV_ALLOCS; i++) {
#if defined(UMPIRE_ENABLE_CUDA)
    allocate_and_use<cuda_platform> cuda;
    ASSERT_NO_THROW(cuda.test(allocator, umpire::UMPIRE_DEV_ALLOCS_h[i].getName()));
#elif defined(UMPIRE_ENABLE_HIP)
    allocate_and_use<hip_platform> hip;
    ASSERT_NO_THROW(hip.test(allocator, umpire::UMPIRE_DEV_ALLOCS_h[i].getName()));
#else
    FAIL(); // If neither CUDA nor HIP is enabled, yet we are testing the DeviceAllocator, something is wrong!
#endif
  }

#if defined(UMPIRE_ENABLE_CUDA)
  allocate_and_use<cuda_platform> cuda;
  ASSERT_THROW(cuda.test(allocator, da1.getName()), umpire::util::Exception);
  ASSERT_NO_THROW(da1.reset());
  ASSERT_NO_THROW(cuda.test(allocator, da1.getName()));
#elif defined(UMPIRE_ENABLE_HIP)
  allocate_and_use<hip_platform> hip;
  ASSERT_THROW(hip.test(allocator, da1.getName()), umpire::util::Exception);
  ASSERT_NO_THROW(da1.reset());
  ASSERT_NO_THROW(hip.test(allocator, da1.getName()));
#endif

  ASSERT_NO_THROW(umpire::destroy_device_allocator());
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);

  int result = RUN_ALL_TESTS();

  return result;
}
