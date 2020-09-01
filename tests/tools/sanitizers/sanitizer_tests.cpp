//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPoolList.hpp"
#include "umpire/strategy/DynamicPoolMap.hpp"

void test_read_after_free()
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("test_allocator");

  const std::size_t SIZE = 1356;
  const std::size_t INDEX = SIZE / 2;
  double* data =
      static_cast<double*>(allocator.allocate(SIZE * sizeof(double)));

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
  double* data =
      static_cast<double*>(allocator.allocate(SIZE * sizeof(double)));

  data[INDEX] = 100;
  std::cout << "data[INDEX] = " << data[INDEX] << std::endl;

  allocator.deallocate(data);
  data[INDEX] = -1;
  std::cout << "data[INDEX] = " << data[INDEX] << std::endl;
}

int main(int argc, char* argv[])
{
  if (argc < 3) {
    std::cout << argv[0]
              << " requires 2 arguments, test type and allocation strategy"
              << std::endl;
  }

  const std::string strategy{argv[1]};
  const std::string test_type{argv[2]};

  auto& rm = umpire::ResourceManager::getInstance();

  if (strategy.find("DynamicPoolMap") != std::string::npos) {
    auto pool = rm.makeAllocator<umpire::strategy::DynamicPoolMap>(
        "test_allocator", rm.getAllocator("HOST"));
    UMPIRE_USE_VAR(pool);
  } else if (strategy.find("DynamicPoolList") != std::string::npos) {
    auto pool = rm.makeAllocator<umpire::strategy::DynamicPoolList>(
        "test_allocator", rm.getAllocator("HOST"));
    UMPIRE_USE_VAR(pool);
  }

  if (test_type.find("read") != std::string::npos) {
    test_read_after_free();
  } else if (test_type.find("write") != std::string::npos) {
    test_write_after_free();
  }

  return 0;
}