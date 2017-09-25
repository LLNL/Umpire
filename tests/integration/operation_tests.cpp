#include "gtest/gtest.h"

#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"

TEST(Operation, HostToHostCopy)
{
  auto rm = umpire::ResourceManager::getInstance();

  umpire::Allocator host_allocator = rm.getAllocator("HOST");

  double* array_one = static_cast<double*>(host_allocator.allocate(100*sizeof(double)));
  double* array_two = static_cast<double*>(host_allocator.allocate(100*sizeof(double)));

  for (int i = 0; i < 100; i++) {
    array_one[i] = i;
  }

  rm.copy(array_one, array_two);

  for (int i = 0; i < 100; i++) {
    ASSERT_DOUBLE_EQ(array_one[i], array_two[i]);
  }
}
