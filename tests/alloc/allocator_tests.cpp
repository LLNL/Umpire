#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

#include "gtest/gtest.h"

static void* allocate_with_allocator(umpire::Allocator* alloc) {
  return alloc->allocate(10 * sizeof(double));
}

TEST(Allocator, ResourceManager) {
  umpire::ResourceManager rm = umpire::ResourceManager::getInstance();

  double* test = static_cast<double*>(allocate_with_allocator(&rm));

  ASSERT_NE(test, nullptr);
}
