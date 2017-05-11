#include "gtest/gtest.h"

#include "umpire/ResourceManager.hpp"

TEST(ResourceManager, Constructor) {
  umpire::ResourceManager rm = umpire::ResourceManager::getInstance();
  SUCCEED();
}
