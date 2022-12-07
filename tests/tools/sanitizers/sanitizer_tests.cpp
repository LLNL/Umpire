//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPoolList.hpp"
#include "umpire/strategy/QuickPool.hpp"

__global__ void test_for_hip(double** data_ptr, std::size_t INDEX)
{
  if (threadIdx.x == 0) {
    *data_ptr[INDEX] = 100;
  }
}

void sanitizer_test(const std::string test_type, const std::string resource_type)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("test_allocator");

  const std::size_t SIZE = 1356;
  const std::size_t INDEX = SIZE / 2;
  double** data = static_cast<double**>(allocator.allocate(SIZE * sizeof(double*)));

  if (test_type.find("read") != std::string::npos) {
    if (resource_type != "HOST") {
      hipLaunchKernelGGL(test_for_hip, dim3(1), dim3(16), 0, 0, data, INDEX);
      hipDeviceSynchronize();
    } else {
      *data[INDEX] = 100;
      std::cout << "data[INDEX] = " << *data[INDEX] << std::endl;
    }

    // Test read after free from host
    allocator.deallocate(data);
    std::cout << "data[256] = " << *data[256] << std::endl;
  } else {
    if (test_type.find("write") == std::string::npos) {
      std::cout << "Test type did not match either option - using write" << std::endl;
    }
    if (resource_type != "HOST") {
      hipLaunchKernelGGL(test_for_hip, dim3(1), dim3(16), 0, 0, data, INDEX);
      hipDeviceSynchronize();
    } else {
      *data[INDEX] = 100;
      std::cout << "data[INDEX] = " << *data[INDEX] << std::endl;
    }

    // Test write after free from host
    allocator.deallocate(data);
    *data[INDEX] = -1;
    std::cout << "data[INDEX] = " << *data[INDEX] << std::endl;
  }
}

int main(int argc, char* argv[])
{
  if (argc < 4) {
    std::cout << "Usage: requires 3 arguments." << std::endl;
    std::cout << "First, an allocation strategy (QuickPool or DynamicPoolList)." << std::endl;
    std::cout << "Second, a test type (read or write)." << std::endl;
    std::cout << "Third, a resource type (DEVICE, HOST, or UM)." << std::endl;
    return 0;
  }

  std::string strategy{argv[1]};
  std::string test_type{argv[2]};
  std::string resource_type{argv[3]};

  auto& rm = umpire::ResourceManager::getInstance();

  if ((resource_type.find("DEVICE") != std::string::npos) || (resource_type.find("UM") != std::string::npos)) {
#if !defined(UMPIRE_ENABLE_HIP)
    UMPIRE_ERROR(runtime_error,
                 umpire::fmt::format("The resource, \"{}\", can't be used if HIP is not enabled.", resource_type));
#endif
  } else {
    if (resource_type.find("HOST") == std::string::npos) {
      std::cout << "Resource type did not match any available options - using HOST." << std::endl;
      resource_type = "HOST";
    }
  }

  if (strategy.find("DynamicPoolList") != std::string::npos) {
    auto pool = rm.makeAllocator<umpire::strategy::DynamicPoolList>("test_allocator", rm.getAllocator(resource_type));
    UMPIRE_USE_VAR(pool);
  } else {
    if (strategy.find("QuickPool") == std::string::npos) {
      std::cout << "Allocation strategy did not match either option - using QuickPool." << std::endl;
      strategy = "QuickPool";
    }
    auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("test_allocator", rm.getAllocator(resource_type));
    UMPIRE_USE_VAR(pool);
  }

  std::cout << " Conducting sanitizer test with " << strategy << " strategy, " << test_type << " test type, and the "
            << resource_type << " resource." << std::endl;

  // Conduct the test
  sanitizer_test(test_type, resource_type);

  return 0;
}
