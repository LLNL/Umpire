//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <stdio.h>

#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPoolList.hpp"
#include "umpire/strategy/QuickPool.hpp"

__global__ void test_read_after_free(double** data_ptr, std::size_t INDEX)
{
  if (threadIdx.x == 0) {
    *data_ptr[INDEX] = 100;
    printf("data_ptr[INDEX] = %f", *data_ptr[INDEX]);
    printf("data_ptr[256] = %f", *data_ptr[256]);
  }
}

void test_read_after_free()
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("test_allocator");

  const std::size_t SIZE = 1356;
  const std::size_t INDEX = SIZE / 2;
  double* data = static_cast<double*>(allocator.allocate(SIZE * sizeof(double)));

  data[INDEX] = 100;
  std::cout << "data[INDEX] = " << data[INDEX] << std::endl;

  allocator.deallocate(data);
  std::cout << "data[256] = " << data[256] << std::endl;
}

void test_write_after_free()
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("test_allocator");

  const std::size_t SIZE = 1356;
  const std::size_t INDEX = SIZE / 2;
  double* data = static_cast<double*>(allocator.allocate(SIZE * sizeof(double)));

  data[INDEX] = 100;
  std::cout << "data[INDEX] = " << data[INDEX] << std::endl;

  allocator.deallocate(data);
  data[INDEX] = -1;
  std::cout << "data[INDEX] = " << data[INDEX] << std::endl;
}

int main(int argc, char* argv[])
{
  if (argc < 3) {
    std::cout << argv[0] << " requires 2 arguments, test type and allocation strategy" << std::endl;
  }

  const std::string strategy{argv[1]};
  const std::string test_type{argv[2]};

#if defined(UMPIRE_ENABLE_HIP)
  const std::string resource_type{"UM"};
#else
  const std::string resource_type{"HOST"};
#endif

  auto& rm = umpire::ResourceManager::getInstance();

  if (strategy.find("QuickPool") != std::string::npos) {
    auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("test_allocator", rm.getAllocator(resource_type));
    UMPIRE_USE_VAR(pool);
  } else if (strategy.find("DynamicPoolList") != std::string::npos) {
    auto pool = rm.makeAllocator<umpire::strategy::DynamicPoolList>("test_allocator", rm.getAllocator(resource_type));
    UMPIRE_USE_VAR(pool);
  }

  if (test_type.find("read") != std::string::npos) {
    //test_read_after_free();
    #if defined(UMPIRE_ENABLE_HIP)
      auto allocator = rm.getAllocator("test_allocator");
      const std::size_t SIZE = 1356;
      const std::size_t INDEX = SIZE / 2;
      double** ptr_to_data = static_cast<double**>(allocator.allocate(sizeof(double*)));
      hipLaunchKernelGGL(test_read_after_free, dim3(1), dim3(16), 0, 0, ptr_to_data, INDEX);

      // Test read after free from host
      allocator.deallocate(ptr_to_data);
      std::cout << "data[256] = " << *ptr_to_data[256] << std::endl;
    #endif
  } else if (test_type.find("write") != std::string::npos) {
    test_write_after_free();
    //#if defined(UMPIRE_ENABLE_HIP)
    //  const std::size_t SIZE = 1356;
    //  const std::size_t INDEX = SIZE / 2;
    //  double** ptr_to_data = static_cast<double**>(allocator.allocate(sizeof(double*)));
    //  hipLaunchKernelGGL(test_write_after_free, dim3(1), dim3(16), 0, 0, ptr_to_data, INDEX);
    //#endif
  }

  return 0;
}
