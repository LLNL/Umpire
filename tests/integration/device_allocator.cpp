//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"
#include "umpire/DeviceAllocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/device_allocator_helper.hpp"

constexpr double NUM = 42.0 * 42.0;
static char* device_allocator_names_h[3] = {(char*)"da1", (char*)"da2", (char*)"da3"};
__device__ static char* device_allocator_names[3] = {(char*)"da1", (char*)"da2", (char*)"da3"};

__global__ void tester_by_name(double** data_ptr, int index)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx == 0) {
    umpire::DeviceAllocator da = umpire::get_device_allocator(device_allocator_names[index]);
    double* data = static_cast<double*>(da.allocate(1 * sizeof(double)));
    *data_ptr = data;
    data[0] = NUM;
  }
}

__global__ void tester_by_ID(double** data_ptr, int index)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx == 10) {
    umpire::DeviceAllocator da = umpire::get_device_allocator(index);
    double* data = static_cast<double*>(da.allocate(1 * sizeof(double)));
    *data_ptr = data;
    data[0] = NUM * 42.0;
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

  ASSERT_NO_THROW((UMPIRE_SET_UP_DEVICE_ALLOCATORS()));
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
  tester_by_name<<<1, 16>>>(data_ptr, str_index);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    UMPIRE_ERROR(umpire::runtime_error,
                 fmt::format("Error when trying to sync CUDA Device: {}", cudaGetErrorString(err)));
  }
#elif defined(UMPIRE_ENABLE_HIP)
  hipLaunchKernelGGL(tester_by_name, dim3(1), dim3(16), 0, 0, data_ptr, str_index);
  hipError_t err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    UMPIRE_ERROR(umpire::runtime_error,
                 fmt::format("Error when trying to sync HIP Device: {}", hipGetErrorString(err)));
  }
#endif

  ASSERT_EQ(*data_ptr[0], NUM);

  auto my_da = umpire::get_device_allocator(GetParam());
  ASSERT_NO_THROW(my_da.reset());

  ASSERT_EQ(my_da.getCurrentSize(), 0);

#if defined(UMPIRE_ENABLE_CUDA)
  tester_by_ID<<<1, 16>>>(data_ptr, str_index);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    UMPIRE_ERROR(umpire::runtime_error,
                 fmt::format("Error when trying to sync CUDA Device: {}", cudaGetErrorString(err)));
  }
#elif defined(UMPIRE_ENABLE_HIP)
  hipLaunchKernelGGL(tester_by_ID, dim3(1), dim3(16), 0, 0, data_ptr, my_da.getID());
  err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    UMPIRE_ERROR(umpire::runtime_error,
                 fmt::format("Error when trying to sync HIP Device: {}", hipGetErrorString(err)));
  }
#endif

  ASSERT_EQ(my_da.getCurrentSize(), sizeof(double));
}

INSTANTIATE_TEST_SUITE_P(DeviceAllocatorTests, DeviceAllocator, ::testing::ValuesIn(device_allocator_names_h));
