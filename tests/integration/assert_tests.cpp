#include "gtest/gtest.h"

#include "umpire/ResourceManager.hpp"
#include "umpire/resource/MemoryResourceTypes.hpp"

TEST(AllocatorAssert, DeallocateDeath)
{
#if !defined(NDEBUG)
  ASSERT_DEATH(umpire::ResourceManager::getInstance().getAllocator("HOST").deallocate( nullptr ), "");
#else
  SUCCEED();
#endif
}

