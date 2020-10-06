#include "umpire/ResourceManager.hpp"

#include "gtest/gtest.h"

// Needs to be in separate file so that resources are not initialized prior to
// reallocate call
TEST(Reallocate, Nullptr)
{
  auto& rm = umpire::ResourceManager::getInstance();
  constexpr std::size_t size = 1024;

  void* ptr{nullptr};
  EXPECT_NO_THROW({
    ptr = rm.reallocate(ptr, size);
  });

  ASSERT_NE(nullptr, ptr);

  rm.deallocate(ptr);
}
