#include "gtest/gtest.h"

#include "umpire/space/CnmemPoolFactory.hpp"
#include "umpire/space/CnmemPool.hpp"

#include "umpire/util/Exception.hpp"

TEST(CnmemPool, Constructor)
{
  umpire::space::MemorySpace* space = new umpire::space::CnmemPool("CNMEM");

  SUCCEED();
}

TEST(CnmemPool, Allocate)
{
  umpire::space::MemorySpace* space = new umpire::space::CnmemPool("CNMEM");

  void* test = space->allocate(100);

  ASSERT_NE(nullptr, test);

  space->free(test);
}
