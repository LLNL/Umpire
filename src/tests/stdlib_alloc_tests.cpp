#include "gtest/gtest.h"

#include "umpire/Umpire.hpp"

TEST(stdlib, malloc_free) {
  double* test = static_cast<double*>(umpire::malloc(5 * sizeof(double)));

  ASSERT_NE(test, nullptr);

  umpire::free(test);
}

